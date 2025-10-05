# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os.path import basename, splitext
from pathlib import Path

import nemo_run as run

from megatron.core.distributed import DistributedDataParallelConfig
from nemo.collections.llm.recipes.qwen3_8b import qwen3_model
from nemo.collections.llm.recipes.llama3_8b import model
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
# from nemo.collections.llm.recipes.llama3_8b import finetune_recipe, model
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin, Optional, PerfEnvPlugin
from nemo.utils import logging
from nemo.utils.exp_manager import TimingCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

# TODO can we be indepentend of?
from nemo.collections.llm.recipes.precision.mixed_precision import (
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
)
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe

from scripts.performance.argument_parser import parse_cli_args
from scripts.performance.helpers import get_user_configs
from scripts.performance.executors import local_executor
from scripts.performance.utils import hf_tokenizer, import_ckpt_experiment, isfile_train_pack_metadata, get_comm_overlap_callback_idx
from scripts.nvfp4fw_mxfp8bw.config_visualizer import ConfigVisualizer
HF_MODEL_URI = "meta-llama/Meta-Llama-3-8B"
# HF_MODEL_URI = "Qwen/Qwen3-8B"

# Set this to True if checkpoint is available at 'NEMO_HOME'. If set to False,
# extra Slurm job will be scheduled. In this case, if checkpoint is available
# at 'NEMO_HOME', fine-tuning job will use this checkpoint, else, it will be
# downloaded from HuggingFace
SKIP_IMPORT = False


# def override_recipe_configs(
#     args: str,
#     num_nodes: int,
#     mbs: int,
#     gbs: int,
#     tp_size: int,
#     pp_size: int,
#     cp_size: int,
#     vp_size: int,
#     ep_size: int,
#     enable_cuda_graphs: bool,
# ):
#     """
#     llama3 8b pre-train recipe aimed at achieving best possible performance.

#     NOTE: Use fp8 precision training with caution. It might not give desirable results.
#     """
#     finetuning_scheme = "none" if args.finetuning == "sft" else args.finetuning

#     gpu_type = args.gpu.lower()
#     if gpu_type in ["b200", "gb200"]:
#         recipe = finetune_recipe(peft_scheme=finetuning_scheme, performance_mode=True, seq_length=16384)
#     else:
#         recipe = finetune_recipe(peft_scheme=finetuning_scheme, performance_mode=True)

#     recipe = set_primary_perf_configs(
#         recipe,
#         finetuning_scheme,
#         num_nodes,
#         args.gpus_per_node,
#         mbs,
#         gbs,
#         args.max_steps,
#         tp_size,
#         pp_size,
#         cp_size,
#         vp_size,
#         ep_size,
#         enable_cuda_graphs=enable_cuda_graphs,
#         compute_dtype=args.compute_dtype,
#         fp8_recipe=args.fp8_recipe,
#         nccl_communicator_config_path=args.nccl_communicator_config_path,
#         use_user_buffer_registration=args.use_user_buffer_registration,
#         use_sharp=args.use_sharp,
#     )
#     recipe = set_exp_logging_configs(
#         recipe,
#         finetuning_scheme,
#         "llm",
#         "llama3",
#         args.tensorboard,
#         args.wandb,
#         args.wandb_prj_name,
#         args.wandb_job_name,
#     )

#     # data module configs
#     if args.use_hf_tokenizer:
#         recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
#     else:
#         recipe.data.tokenizer = run.Config(
#             get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=128256
#         )
#         recipe.model.tokenizer = recipe.data.tokenizer
#     if recipe.data.__fn_or_cls__ == SquadDataModule and not isfile_train_pack_metadata(HF_MODEL_URI, recipe.data):
#         # flag is valid only for SquadDataModule
#         recipe.data.force_redownload = True

#     recipe.optim.config.use_distributed_optimizer = True
#     recipe.model.config.disable_parameter_transpose_cache = True

#     return recipe

SEQUENCE_LENGTH = 4096
MBS = 1
GBS = 512
TRAIN_STEPS = 200
VAL_INTERVAL = 50

if __name__ == "__main__":
    args = parse_cli_args().parse_args()

    if args.wandb:
        assert args.wandb_key is not None, "wandb logger needs \"wandb_key\""
        assert args.wandb_prj_name is not None, "wandb logger needs \"wandb_prj_name\""
        assert args.wandb_job_name is not None, "wandb logger needs \"wandb_job_name\""

    kwargs = get_user_configs(args.gpu.lower(), args.finetuning, "llama3", "8b", args) # since qwen3_8b is similar to llama3_8b, we use llama3_8b config
    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, _, enable_cuda_graphs = kwargs[:10]

    finetuning_scheme = "none" if args.finetuning == "sft" else args.finetuning

    gpu_type = args.gpu.lower()
    num_gpus_per_node = args.gpus_per_node
    max_steps = args.max_steps

    # recipe default initialization =================================================================
    # TODO: do we really need this?
    if gpu_type in ["b200", "gb200"]:
        seq_length = 16384

    dir = None
    name = "default"

    peft_scheme=finetuning_scheme
    performance_mode=True
    packed_sequence = performance_mode

    # For unpacked sequence, most samples in SQuAD dataset are shorter than 2K
    if seq_length is None:
        seq_length = 4096 if packed_sequence else 2048

    if HF_MODEL_URI == "meta-llama/Meta-Llama-3-8B":
        model_cfg = model()
    elif HF_MODEL_URI == "Qwen/Qwen3-8B":
        model_cfg = qwen3_model(version="qwen3_8b")
    else:
        raise ValueError(f"Unrecognized model uri: {HF_MODEL_URI}")
    
    recipe = default_finetune_recipe(
        model_cfg, HF_MODEL_URI, dir, name, num_nodes, num_gpus_per_node, packed_sequence
    )

    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 2
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() in ['lora', 'dora']:
        recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
        recipe.peft.dim = 8
        recipe.peft.alpha = 16
        recipe.optim.config.use_distributed_optimizer = False
        # some settings currently do not function correctly with LoRA
        recipe.model.config.cross_entropy_loss_fusion = False
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    # Sequence length settings in the model and dataset must agree
    recipe.model.config.seq_length = seq_length
    recipe.data.seq_length = seq_length
    if packed_sequence:
        recipe.data.dataset_kwargs = {'pad_to_max_length': True}
        recipe.data.packed_sequence_specs = run.Config(PackedSequenceSpecs, packed_sequence_size=seq_length)

    if performance_mode:
        # recipe = finetune_performance_optimizations(recipe, peft_scheme)
        recipe.trainer.strategy.tensor_model_parallel_size = 1

        if not hasattr(recipe.trainer, "callbacks") or recipe.trainer.callbacks is None:
            recipe.trainer.callbacks = []

        if peft_scheme is None or peft_scheme.lower() == 'none':
            recipe.trainer.strategy.ddp = run.Config(
                DistributedDataParallelConfig,
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=False,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=True,
            )
        else:
            recipe.peft.target_modules = ['linear_qkv']

        recipe.trainer.plugins.grad_reduce_in_fp32 = False

        recipe.trainer.callbacks.append(
            run.Config(
                MegatronCommOverlapCallback,
                tp_comm_overlap=False,
            )
        )
        recipe.trainer.callbacks.append(run.Config(TimingCallback))
        recipe.trainer.callbacks.append(
            run.Config(
                GarbageCollectionCallback,
                100,
                100,
            )
        )

        recipe.optim.config.use_precision_aware_optimizer = False
    # =================================================================
    task = finetuning_scheme
    
    etp_size = None
    use_mcore_fsdp = False
    use_fsdp_double_buffer = False
    use_user_buffer_registration = args.use_user_buffer_registration
    use_sharp = args.use_sharp
    recompute_layers = 0
    activation_offload_layers = 0
    compute_dtype=args.compute_dtype
    fp8_recipe=args.fp8_recipe
    recompute_modules = None
    nccl_communicator_config_path = args.nccl_communicator_config_path
    keep_fsdp_fp8_transpose_cache = None

    recipe.trainer.num_nodes = num_nodes
    recipe.trainer.devices = num_gpus_per_node
    recipe.trainer.max_steps = max_steps
    recipe.trainer.limit_val_batches = 32

    # lightning.pytorch.LightningDataModule configs
    recipe.data.micro_batch_size = mbs
    recipe.data.global_batch_size = gbs
    if recipe.data.__fn_or_cls__ == MockDataModule:
        recipe.data.num_train_samples = max_steps * gbs  # ensure only 1 epoch for whole run

    # parallelism configs
    recipe.trainer.strategy.tensor_model_parallel_size = tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = pp_size
    recipe.trainer.strategy.context_parallel_size = cp_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None if vp_size == 1 else vp_size
    recipe.trainer.strategy.expert_model_parallel_size = ep_size
    recipe.trainer.strategy.expert_tensor_parallel_size = etp_size
    recipe.trainer.strategy.sequence_parallel = bool(tp_size > 1)
    if nccl_communicator_config_path is not None:
        recipe.trainer.strategy.nccl_communicator_config_path = nccl_communicator_config_path

    # callback configs
    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)
    if comm_overlap_callback_idx is not None:
        # WARNING: If True, checkpointing (if enabled) might not work
        recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather_with_optimizer_step = bool(
            dp_size > 1 and pp_size > 1 and vp_size and vp_size > 1
        )

    # ==== performance optimization configs
    # recipe = set_perf_optimization_configs(
    #     recipe=recipe,
    #     use_mcore_fsdp=use_mcore_fsdp,
    #     enable_cuda_graphs=enable_cuda_graphs,
    #     task=task,
    #     tp_size=tp_size,
    #     compute_dtype=compute_dtype,
    #     fp8_recipe=fp8_recipe,
    #     recompute_layers=recompute_layers,
    #     activation_offload_layers=activation_offload_layers,
    #     recompute_modules=recompute_modules,
    #     use_fsdp_double_buffer=use_fsdp_double_buffer,
    #     use_user_buffer_registration=use_user_buffer_registration,
    #     use_sharp=use_sharp,
    #     keep_fsdp_fp8_transpose_cache=keep_fsdp_fp8_transpose_cache,
    # )
    recipe.model.config.cross_entropy_fusion_impl = "te"

    if use_fsdp_double_buffer:
        assert use_mcore_fsdp == True, "use_fsdp_double_buffer requires use_mcore_fsdp to be True"

    if use_user_buffer_registration:
        assert use_mcore_fsdp == True, "use_user_buffer_registration requires use_mcore_fsdp to be True"
        assert (
            use_fsdp_double_buffer is not False
        ), "use_fsdp_double_buffer cannot be False when use_user_buffer_registration is True"

    if use_mcore_fsdp and enable_cuda_graphs:
        logging.warning("Currently, cuda graphs are not supported with FSDP. Disabling cuda graphs.")
        enable_cuda_graphs = False

    # === set_cuda_graph_configs =================
    # recipe = set_cuda_graph_configs(recipe, enable_cuda_graphs, task)
    recipe.model.config.enable_cuda_graph = enable_cuda_graphs
    recipe.trainer.strategy.use_te_rng_tracker = enable_cuda_graphs
    if (
        task in ["none", "lora"]
        and hasattr(recipe.data, "packed_sequence_specs")
        and recipe.data.packed_sequence_specs is not None
    ):
        recipe.data.packed_sequence_specs.pad_cu_seqlens = enable_cuda_graphs

    if use_mcore_fsdp:
        # comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
        if recipe.trainer.callbacks:  # default is None in lightning
            for idx, callback in enumerate(recipe.trainer.callbacks):
                if callback.__fn_or_cls__ == MegatronCommOverlapCallback:
                    comm_overlap_callback_idx = idx

        # === set_mcore_fsdp_configs ===
        # recipe = set_mcore_fsdp_configs(recipe, comm_overlap_callback_idx, tp_size)
        recipe.model.config.init_model_with_meta_device = True
        recipe.trainer.strategy.fsdp = "megatron"
        recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        # At fp32 gradient, `recipe.trainer.strategy.ddp.gradient_reduce_div_fusion` is used for fusion
        if recipe.trainer.plugins.grad_reduce_in_fp32:
            recipe.trainer.strategy.ddp.average_in_collective = False
        recipe.trainer.strategy.ddp.keep_fp8_transpose_cache_when_using_custom_fsdp = False
        recipe.model.config.gradient_accumulation_fusion = False
        if (
            comm_overlap_callback_idx is not None
            and recipe.trainer.callbacks[comm_overlap_callback_idx].defer_embedding_wgrad_compute
        ):
            logging.warning("Disabling deferring embedding wgrad compute because it cannot work with FSDP together.")
            recipe.trainer.callbacks[comm_overlap_callback_idx].defer_embedding_wgrad_compute = False

    #  === set_precision_configs =================
    # recipe = set_precision_configs(recipe, compute_dtype, fp8_recipe)
    if compute_dtype is not None:
        if compute_dtype.lower() == "bf16":
            recipe.optim.config.use_precision_aware_optimizer = True

        if compute_dtype is not None and compute_dtype.lower() == "fp8":
            if fp8_recipe is None:
                fp8_recipe = "ds"
            if fp8_recipe.lower() == "ds":
                recipe.trainer.plugins = bf16_with_fp8_mixed()
            elif fp8_recipe.lower() == "cs":
                recipe.trainer.plugins = bf16_with_fp8_current_scaling_mixed()
                # disable first/last layer bf16 for benchmarking
                recipe.trainer.plugins.first_last_layers_bf16 = False
            elif fp8_recipe.lower() == "mxfp8":
                recipe.trainer.plugins = bf16_with_mxfp8_mixed()
            elif fp8_recipe.lower() == "ss":
                recipe.trainer.plugins = bf16_with_fp8_subchannel_scaling_mixed()

        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    # Enable reuse_grad_buf_for_mxfp8_param_ag for MXFP8 and disable AG overlap
    # because it is not supported with reuse_grad_buf_for_mxfp8_param_ag
    if compute_dtype.lower() == "fp8" and fp8_recipe.lower() == "mxfp8":
        comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
        if comm_overlap_callback_idx is not None:
            recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather = False
        logging.warning(
            "When using MXFP8, to reduce memory usage, we use reuse_grad_buf_for_mxfp8_param_ag. "
            "Disabling AG overlap because it is not supported with reuse_grad_buf_for_mxfp8_param_ag."
        )

    # === set_recompute_configs =================
    # recipe = set_recompute_configs(recipe, recompute_layers, activation_offload_layers, recompute_modules)
    if recompute_layers > 0:
        recipe.model.config.recompute_granularity = "full"
        recipe.model.config.recompute_method = "block"
        recipe.model.config.recompute_num_layers = recompute_layers

    # Activation cpu offloading 
    if activation_offload_layers > 0:
        recipe.model.config.cpu_offloading = True
        recipe.model.config.cpu_offloading_weights = False
        recipe.model.config.cpu_offloading_num_layers = activation_offload_layers

    # Activation recompute configs
    if recompute_modules is not None:
        recipe.model.config.recompute_modules = recompute_modules
        assert (
            recipe.model.config.recompute_granularity == "selective"
        ), "recompute_granularity must be selective when recompute_modules is provided"
        assert (
            recipe.model.config.recompute_num_layers is None
        ), "recompute_num_layers must be None when recompute_modules is provided"
    # end of set_recompute_configs ============================
    recipe.trainer.strategy.use_sharp = bool(use_sharp)

    is_ddp_obj = hasattr(recipe.trainer.strategy, "ddp") and not isinstance(recipe.trainer.strategy.ddp, str)
    if use_user_buffer_registration and not is_ddp_obj:
        logging.warning("DDP is not configured. Cannot use user buffer registration.")
    if is_ddp_obj:
        # Disable local gradient checker at non-debugging mode
        recipe.trainer.strategy.ddp.check_for_nan_in_grad = False
        recipe.trainer.strategy.ddp.check_for_large_grads = False
        recipe.trainer.strategy.ddp.nccl_ub = bool(use_user_buffer_registration)
        recipe.trainer.strategy.ddp.fsdp_double_buffer = bool(use_fsdp_double_buffer)
        recipe.trainer.strategy.ddp.keep_fp8_transpose_cache_when_using_custom_fsdp = bool(
            keep_fsdp_fp8_transpose_cache
        )
    # ==== performance optimization configs ==== End of

    # ==== set_exp_logging_configs
    # recipe = set_exp_logging_configs(
    #     recipe,
    #     finetuning_scheme,
    #     "llm",
    #     "llama3",
    #     args.tensorboard,
    #     args.wandb,
    #     args.wandb_prj_name,
    #     args.wandb_job_name,
    # )

    domain="llm"
    model_name="llama3"
    enable_tb=args.tensorboard
    enable_wd=args.wandb
    wandb_prj_name=args.wandb_prj_name
    wandb_job_name=args.wandb_job_name

    if task == "pre_train" and domain == "llm":
        recipe.trainer.callbacks.append(
            run.Config(
                FLOPsMeasurementCallback,
                model_config=recipe.model.config,
                data_config=recipe.data,
                model_name=model_name,
            )
        )

    if not enable_tb:  # tensorboard adds performance overhead.
        recipe.log.tensorboard = None
        # recipe.trainer.logger = False
    else:
        # default path is NOT intuitive- `<log_dir>/code/nemo_experiments/tb_logs/default/<tfevents_file>`
        recipe.log.log_dir = "/nemo_run/lightning_logs"  # saves file at- `<log_dir>/lightning_logs/tb_logs
    if enable_wd:
        from nemo.collections.llm.recipes.log.default import wandb_logger

        recipe.log.wandb = wandb_logger(project=wandb_prj_name, name=wandb_job_name)

    # Misc. for overall faster experiment runtime
    recipe.log.ckpt = None

    # disable checkpointing if no ModelCheckpoint callback is found
    callbacks = recipe.trainer.callbacks
    checkpoint_callback_idx = None
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if callback.__fn_or_cls__ == ModelCheckpoint:
                checkpoint_callback_idx = idx
                break
    recipe.trainer.enable_checkpointing = checkpoint_callback_idx is not None
    recipe.trainer.log_every_n_steps = 1
    # ================= eod et_exp_logging_configs

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=128256
        )
        recipe.model.tokenizer = recipe.data.tokenizer
    if recipe.data.__fn_or_cls__ == SquadDataModule and not isfile_train_pack_metadata(HF_MODEL_URI, recipe.data):
        # flag is valid only for SquadDataModule
        recipe.data.force_redownload = True

    recipe.optim.config.use_distributed_optimizer = True
    recipe.model.config.disable_parameter_transpose_cache = True

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # exp_config = f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{mbs}mbs_{gbs}gbs"
    # exp_name = f"{args.finetuning}_{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"
    exp_config = f"{args.num_gpus}gpus"
    exp_name = f"{args.finetuning}_{args.compute_dtype}_{exp_config}"

    executor = local_executor(
        args.gpu.lower(),
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars={},
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
        network='sharp' if args.use_sharp else None,
    )

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
            gpu_sm100_or_newer=(args.gpu.lower() in ['b200', 'gb200']),
        )
    ]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))
    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

    # exp_dir = Path(run.config.get_nemorun_home())
    # viz = ConfigVisualizer(recipe, outdir=exp_dir)
    # viz.print_all()
    # viz.draw()

    with run.Experiment(exp_name) as exp:
        if not (Path(os.getenv("NEMO_HOME",""))/"models"/ HF_MODEL_URI).exists():
            if os.getenv("HF_TOKEN") is None:
                raise ValueError("Model not found, need to download from hf and convert nemo format, pls set/export HF_TOKEN")
            exp.add(*import_ckpt_experiment(executor, model(), source=f"hf://{HF_MODEL_URI}"), tail_logs=True)

        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
            tail_logs=True
        )

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()
