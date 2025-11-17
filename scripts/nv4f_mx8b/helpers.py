import os
import sys
from typing import Dict, List

import nemo_run as run
from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.launcher import SlurmTemplate

from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

import argparse
from nemo_run.config import get_nemorun_home
import wandb

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)

# NOTE: If you update this template,
# PLEASE test it by submitting a job to GPU/node/cluster and verifying the sbatch and bash scripts.
INLINE_TEMPLATE = r"""
#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c '{{ pre_cmds }} {{ command }}'
"""

def import_ckpt_fn(executor: run.SlurmExecutor, model: run.Config, source: str):
    """
    Downloads/Acceses checkpoint to be used for fine-tuning. `import_ckpt` first tries find the nemo checkpoint in
    <NEMO_HOME>/models/. For eg: for llama3 8b, the path will look like- <NEMO_HOME>/models/meta-llama/Meta-Llama-3-8B
    If missing, tries to downloads at the same location from HuggingFace and converts it nemo format.

    Args:
        source (str): HuggingFace URL. For eg- hf://meta-llama/Meta-Llama-3-70B
    """
    from copy import deepcopy

    from nemo.collections.llm import import_ckpt

    import_executor = deepcopy(executor)
    import_executor.ntasks_per_node = 1
    import_executor.nodes = 1

    return run.Partial(import_ckpt, model=model, source=source, overwrite=False), import_executor


def local_executor(
    gpu: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    custom_srun_args: List[str] = [],
    nemo_home: str = DEFAULT_NEMO_HOME,
    custom_bash_cmds: List[str] = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    PERF_ENV_VARS = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FLASH_ATTN": "1",  # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NVTE_FUSED_ATTN": "1",  # Enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1",  # Print memory allocation
    }

    custom_bash_cmds = [] if custom_bash_cmds is None else custom_bash_cmds
    err_msgs = []
    mounts = []
    srun_args = custom_srun_args.copy() + ["--mpi=pmix", "--no-container-mount-home"]

    if gpu.lower() not in ['b200']:
        # TODO: we currently disable PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        # on B200 as it causes an unexpected error. Add back when issue is debugged and fixed.
        PERF_ENV_VARS["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    PERF_ENV_VARS["NEMORUN_HOME"] = log_dir

    if gpu.lower() == 'gb200':
        PERF_ENV_VARS["NCCL_NET_GDR_LEVEL"] = "PHB"  # For NCCL 2.25
        PERF_ENV_VARS["NCCL_NET_GDR_C2C"] = "1"  # For NCCL 2.26

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this to 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        PERF_ENV_VARS["NEMO_HOME"] = nemo_home
        mounts.extend([f"{nemo_home}:{nemo_home}"])

    PERF_ENV_VARS.update({"HF_TOKEN": os.getenv("HF_TOKEN", ""), "TRANSFORMERS_OFFLINE": "0"})
    PERF_ENV_VARS.update({"WANDB_API_KEY": os.getenv("WANDB_API_KEY", "")})
    
    PERF_ENV_VARS |= custom_env_vars
    mounts.extend(custom_mounts)

    # add --segment flag to sbatch if job uses GB200 and goes beyond one rack.
    segment = None
    if num_gpus_per_node == 4 and nodes > 18:
        for segment_candidate in range(18, 0, -1):
            if nodes % segment_candidate == 0:
                segment = segment_candidate
                break

    numa_divisor = 2 if gpu.lower() == 'gb200' else 4
    numa_cmd = f"numactl --cpunodebind=$((SLURM_LOCALID/{numa_divisor})) --membind=$((SLURM_LOCALID/{numa_divisor}))"
    custom_bash_cmds.append(numa_cmd)

    launcher = SlurmTemplate(
        template_inline=INLINE_TEMPLATE,
        template_vars={"pre_cmds": " ; ".join(custom_bash_cmds)},
    )

    executor = run.LocalExecutor(
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        env_vars=PERF_ENV_VARS,
        # launcher=launcher,
    )

    return executor


def is_wandb_logged_in():
    """Check if wandb is configured with an API key"""
    try:
        # Check environment variable
        if os.environ.get("WANDB_API_KEY"):
            return True
        
        # Check wandb settings
        api_key = wandb.api.api_key
        if api_key:
            return True
            
        return False
    except Exception:
        return False


def parse_cli_args():
    """
    Command line arguments correspong to Slurm cluster and NeMo2.0 for running pre-training and
    fine-tuning experiments.
    """
    parser = argparse.ArgumentParser(description="NeMo2.0 Performance Pretraining and Fine-Tuning")

    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        choices=["h100", "b200", "gb200"],
        help="Target gpu type.",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        choices=["bf16", "fp8"],
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    fp8_recipe_msg = (
        "FP8 recipe. Options- ds (per-tensor delayed scaling), cs (per-tensor current scaling), "
        "mxfp8, ss (subchannel scaling). Defaults to ds"
    )
    parser.add_argument(
        "-fr",
        "--fp8_recipe",
        type=str,
        choices=["ds", "cs", "mxfp8", "ss", "nv4f_mx8b"],
        help=fp8_recipe_msg,
        required=False,
        default="ds",
    )
    parser.add_argument(
        "-en",
        "--enable_nsys",
        help="Enable Nsys profiling. Diabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-em",
        "--enable_memory_profile",
        help="Enable memory usage profiling. Diabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-mp",
        "--memory_profile_out_path",
        type=str,
        help="Path to the output file of memory profiling",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdp",
        "--wandb_project",
        type=str,
        help="wandb project name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-f",
        "--finetuning",
        choices=["sft", "lora"],
        help="Finetuning scheme to use. Defaults to 'lora'",
        default='lora',
    )
    nemo_home_msg = [
        "Sets env var `NEMO_HOME` (on compute node using sbatch script)- directory where NeMo searches",
        "for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already",
        f"exist here. Missing files will be downloaded here from HuggingFace. Defaults to {DEFAULT_NEMO_HOME}",
    ]
    parser.add_argument(
        "-nh",
        "--nemo_home",
        type=str,
        help=" ".join(nemo_home_msg),
        default=DEFAULT_NEMO_HOME,
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        help="If true, prints sbatch script to terminal without launching experiment.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-tp",
        "--tensor_parallel_size",
        type=int,
        help="Intra-layer model parallelism. Splits tensors across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-pp",
        "--pipeline_parallel_size",
        type=int,
        help="Inter-layer model parallelism. Splits transformer layers across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cp",
        "--context_parallel_size",
        type=int,
        help="Splits network input along sequence dimension across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-vp",
        "--virtual_pipeline_parallel_size",
        type=int,
        help="Number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ep",
        "--expert_parallel_size",
        type=int,
        help="Distributes Moe Experts across sub data parallel dimension.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-et",
        "--expert_tensor_parallel_size",
        type=lambda x: int(x) if x is not None else None,
        nargs="?",
        const=None,
        help="Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks.\
            Use -et/--expert_tensor_parallel_size <space> for None or -et/--expert_tensor_parallel_size <int>",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-mb",
        "--micro_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gb",
        "--global_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        help="Number of gpus.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gn",
        "--gpus_per_node",
        type=int,
        help="Number of gpus per node. Defaults to 8",
        required=False,
        default=8,
    )
    parser.add_argument(
        "-ms",
        "--max_steps",
        type=int,
        help="Number of train steps. Defaults to 100",
        required=False,
        default=200,
    )
    parser.add_argument(
        "-vms",
        "--val_max_steps",
        type=int,
        help="Number of train steps. Defaults to 100",
        required=False,
        default=32,
    )
    parser.add_argument(
        "-vi",
        "--val_interval",
        type=int,
        help="Number of train steps. Defaults to 100",
        required=False,
        default=50,
    )
    parser.add_argument(
        "-si",
        "--save_interval",
        type=int,
        help="Number of train steps. Defaults to 100",
        required=False,
        default=50,
    )
    def bool_arg(arg):
        if arg.lower() in ['true', '1', 't', 'yes', 'y']:
            return True
        elif arg.lower() in ['false', '0', 'f', 'no', 'n']:
            return False
        else:
            raise ValueError(f"Invalid value for boolean argument: {arg}")

    parser.add_argument(
        "-cg",
        "--cuda_graphs",
        help="Enable CUDA graphs. Disabled by default",
        type=bool_arg,
        required=False,
        default=False,  # NOTE: DO NOT SET DEFAULT TO FALSE, IT WILL BE OVERRIDDEN BY THE RECOMMENDED MODEL CONFIGS
    )
    parser.add_argument(
        "-fsdp",
        "--use_mcore_fsdp",
        help="Enable Megatron Core (Mcore) FSDP. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-fsdp_db",
        "--use_fsdp_double_buffer",
        help="Enable FSDP double buffer. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ubr",
        "--use_user_buffer_registration",
        help="Enable user buffer registration. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-sharp",
        "--use_sharp",
        help="Enable sharp. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-rl",
        "--recompute_layers",
        type=int,
        help="Number of Transformer layers to recompute, where all the intermediate "
        "activations of a Transformer layer are computed. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ol",
        "--activation_offload_layers",
        type=int,
        help="Number of Transformer layers to offload to the CPU memory. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--nccl_communicator_config_path",
        type=str,
        help="Path to NCCL communicator config yaml file",
        required=False,
        default=None,
    )

    def list_of_strings(arg):
        return arg.split(',')

    parser.add_argument(
        "-rm",
        "--recompute_modules",
        nargs="*",
        const=None,
        type=str,
        help="List of modules to perform selective activation recompute. "
        "Users can provide 0 or any number of arguments. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cm",
        "--custom_mounts",
        type=list_of_strings,
        help="Comma separated string of mounts",
        required=False,
        default=[],
    )
    parser.add_argument(
        "--use_hf_tokenizer",
        help="Use HuggingFace tokenizer. Disabled by default. Null tokenizer will be used if not provided.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--keep_fsdp_fp8_transpose_cache",
        help="Keep FSDP FP8 transpose cache. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-dp",
        "--data_path",
        type=str,
        help="Path to preprocessed dataset",
        required=True,
    )
    return parser
