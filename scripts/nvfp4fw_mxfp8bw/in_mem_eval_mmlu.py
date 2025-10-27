# Require TensorRT-Model-Optimizer, add it to PYTHONPATH
from modelopt.torch.utils.plugins.megatron_mmlu import megatron_mmlu
import torch
from nemo import lightning as nl
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec
from nemo.collections.llm.recipes.precision.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
    bf16_with_fw_nvfp4_bw_mxfp8,
)
import fiddle as fdl
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from functools import partial
import argparse

from time import perf_counter
from contextlib import contextmanager, nullcontext

@contextmanager
def timeit(label: str = ""):
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        print(f"{label}{dt/60:.3f} min")

def parse_cli_args():
    parser = argparse.ArgumentParser(description="evaluate NeMo model/ckpt on MMLU")
    parser.add_argument(
        "--nemo_ckpt",
        type=str,
        help="Path to NeMo checkpoint/model",
        required=True,
    )
    precision_msg = (
        "precision mode (using Transformer Engine as backend). Options fp8 (per-tensor fp8), mxfp8, defaults to None (BF16 only)."
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        choices=["fp8", "mxfp8", "nv4f_mx8b"],
        help=precision_msg,
        required=False,
        default=None,
    )
    return parser

if __name__ == "__main__":
        
    args = parse_cli_args().parse_args()
    # model_path = "/root/work/nemo/models/Qwen/Qwen3-8B"
    # model_path = "/root/work/run/mon-nv4f_mx8b/experiments/Qwen3-8B/Qwen3-8B_1761014654/02_Qwen3-8B_sft_f8_nv4f_mx8b/default/2025-10-21_02-44-29/checkpoints/model_name=0--val_loss=0.00-step=199-consumed_samples=102400.0-last"
    
    model = io.load_context(path=ckpt_to_context_subdir(args.nemo_ckpt), subpath="model")

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_optimizer=False,
        ckpt_parallel_save_optim=False,
        setup_optimizers=False,
        ddp="pytorch",
    )

    # not sure where to get the recipe context yet, so just create a default one
    plugins=fdl.build(bf16_mixed())
    fp8_recipe = None

    if args.precision == 'fp8':
        plugins=fdl.build(bf16_with_fp8_current_scaling_mixed())
        fp8_recipe = recipe.Float8CurrentScaling()
    elif args.precision == 'mxfp8':
        plugins=fdl.build(bf16_with_mxfp8_mixed())
        fp8_recipe = recipe.MXFP8BlockScaling()
    elif args.precision == 'nv4f_mx8b':
        plugins=fdl.build(bf16_with_fw_nvfp4_bw_mxfp8())
        fp8_recipe = recipe.NVFP4FwdMXFP8BwdScaling()

    def get_fp8_context():
        # trainer.precision_plugin.forward_context() doesn't return the context
        return te.fp8_autocast(fp8_recipe=fp8_recipe) if fp8_recipe is not None else nullcontext()

    trainer = nl.Trainer(
        devices=1,
        num_nodes=1,
        accelerator="gpu",
        strategy=strategy,
    )

    _setup_trainer_and_restore_model(args.nemo_ckpt, trainer, model)

    # monkey patching so that we dont need to patch the actual submodule repo and install one
    # model.tokenizer.tokenizer = partial(model.tokenizer.tokenizer, padding=True, pad_to_multiple_of=8)

    with timeit("mmlu eval elapse: "):
        with torch.inference_mode():
            with get_fp8_context() as ctx:
                megatron_mmlu(model.module, model.tokenizer.tokenizer)

    print("end.")