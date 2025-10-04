#!/usr/bin/env python
# coding: utf-8

# # Learning Goals
# 
# ## Optimizing Foundation Models with Parameter-Efficient Fine-Tuning (PEFT)
# 
# This notebook aims to demonstrate how to adapt or customize foundation models to improve performance on specific tasks using NeMo 2.0.
# 
# This optimization process is known as fine-tuning, which involves adjusting the weights of a pre-trained foundation model with custom data.
# 
# Considering that foundation models can be significantly large, a variant of fine-tuning has gained traction recently known as PEFT. PEFT encompasses several methods, including P-Tuning, LoRA, Adapters, IA3, etc. NeMo 2.0 currently supports [Low-Rank Adaptation (LoRA)](https://arxiv.org/pdf/2106.09685) method.
# 
# NeMo 2.0 introduces Python-based configurations, PyTorch Lightning’s modular abstractions, and NeMo-Run for scaling experiments across multiple GPUs. In this notebook, we will use NeMo-Run to streamline the configuration and execution of our experiments.
# 
# ## Data
# This notebook uses the SQuAD dataset. For more details about the data, refer to [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
# 
# 

# 
# ## Step 1. Import the Hugging Face Checkpoint
# We use the `llm.import_ckpt` API to download the specified model using the "hf://<huggingface_model_id>" URL format. It will then convert the model into NeMo 2.0 format. For all model supported in NeMo 2.0, refer to [Large Language Models](https://docs.nvidia.com/nemo-framework/user-guide/24.09/llms/index.html#large-language-models) section of NeMo Framework User Guide.

# In[ ]:


import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.llm.peft.lora import LoRA
import torch
import lightning.pytorch as pl
from pathlib import Path
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed


# llm.import_ckpt is the nemo2 API for converting Hugging Face checkpoint to NeMo format
# example usage:
# llm.import_ckpt(model=llm.llama3_8b.model(), source="hf://meta-llama/Meta-Llama-3-8B")
#
# We use run.Partial to configure this function
# def configure_checkpoint_conversion():
#     return run.Partial(
#         llm.import_ckpt,
#         model=llm.llama3_8b.model(),
#         source="hf://meta-llama/Meta-Llama-3-8B",
#         overwrite=False,
#     )

# # configure your function
# import_ckpt = configure_checkpoint_conversion()
# # define your executor
# local_executor = run.LocalExecutor()

# # run your experiment
# run.run(import_ckpt, executor=local_executor)


# ## Step 2. Prepare the Data
# 
# We will be using SQuAD for this notebook. NeMo 2.0 already provides a `SquadDataModule`. Example usage:

# In[2]:


def squad() -> run.Config[pl.LightningDataModule]:
    return run.Config(llm.SquadDataModule, seq_length=2048, micro_batch_size=1, global_batch_size=8, num_workers=0)


# To learn how to use your own data to create a custom `DataModule` for performing PEFT, refer to [NeMo 2.0 SFT notebook](./nemo2-sft.ipynb).

# ## Step 3.1: Configure PEFT with NeMo 2.0 API and NeMo-Run
# 
# The following Python script utilizes the NeMo 2.0 API to perform PEFT. In this script, we are configuring the following components for training. These components are similar between SFT and PEFT. SFT and PEFT both use `llm.finetune` API. To switch from SFT to PEFT, you just need to add `peft` with the LoRA adapter to the API parameter.
# 
# ### Configure the Trainer
# The NeMo 2.0 Trainer works similarly to the PyTorch Lightning trainer.
# 

# In[3]:


def trainer() -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=1
    )
    trainer = run.Config(
        nl.Trainer,
        devices=1,
        max_steps=20,
        accelerator="gpu",
        strategy=strategy,
        plugins=bf16_mixed(),
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=2,
        num_sanity_val_steps=0,
    )
    return trainer


# ### Configure the Logger
# Configure your training steps, output directories and logging through `NeMoLogger`. In the following example, the experiment output will be saved at `./results/nemo2_peft`.
# 
# 

# In[4]:


def logger() -> run.Config[nl.NeMoLogger]:
    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        every_n_train_steps=10,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return run.Config(
        nl.NeMoLogger,
        name="nemo2_peft",
        log_dir="./results",
        use_datetime_version=False,
        ckpt=ckpt,
        wandb=None
    )


# 
# 
# ### Configure the Optimizer
# In the following example, we will be using the distributed adam optimizer and pass in the optimizer configuration through `OptimizerConfig`: 

# In[5]:


def adam() -> run.Config[nl.OptimizerModule]:
    opt_cfg = run.Config(
        OptimizerConfig,
        optimizer="adam",
        lr=0.0001,
        adam_beta2=0.98,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True,
    )
    return run.Config(
        nl.MegatronOptimizerModule,
        config=opt_cfg
    )


# ### Pass in the LoRA Adapter
# We need to pass in the LoRA adapter to our fine-tuning API to perform LoRA fine-tuning. We can configure the adapter as follows. The target module we support includes: `linear_qkv`, `linear_proj`, `linear_fc1` and `linear_fc2`. In the final script, we used the default configurations for LoRA (`llm.peft.LoRA()`), which will use the full list with `dim=32`.

# In[6]:


def lora() -> run.Config[nl.pytorch.callbacks.PEFT]:
    return run.Config(LoRA)


# ### Configure the Base Model
# We will perform PEFT on top of Llama-3-8b, so we create a `LlamaModel` to pass to the NeMo 2.0 finetune API.

# In[7]:


def llama3_8b() -> run.Config[pl.LightningModule]:
    return run.Config(llm.LlamaModel, config=run.Config(llm.Llama3Config8B))


# ### Auto Resume
# In NeMo 2.0, we can directly pass in the Llama3-8b Hugging Face ID to start PEFT without manually converting it into the NeMo checkpoint, as required in NeMo 1.0.

# In[8]:


def resume() -> run.Config[nl.AutoResume]:
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig,
            path="nemo://meta-llama/Meta-Llama-3-8B"
        ),
        resume_if_exists=True,
    )


# 
# ### Configure the NeMo 2.0 finetune API
# Using all the components we created above, we can call the NeMo 2.0 finetune API. The python example usage is as below:
# ```
# llm.finetune(
#     model=llama3_8b(),
#     data=squad(),
#     trainer=trainer(),
#     peft=lora(),
#     log=logger(),
#     optim=adam(),
#     resume=resume(),
# )
# ```
# We configure the `llm.finetune` API as below:

# In[ ]:


def configure_finetuning_recipe():
    return run.Partial(
        llm.finetune,
        model=llama3_8b(),
        trainer=trainer(),
        data=squad(),
        log=logger(),
        peft=lora(),
        optim=adam(),
        resume=resume(),
    )


# ## Step 3.2: Run PEFT with NeMo 2.0 API and NeMo-Run
# 
# We use `LocalExecutor` for executing our configured finetune function. For more details on the NeMo-Run executor, refer to [Execute NeMo Run](https://github.com/NVIDIA/NeMo-Run/blob/main/docs/source/guides/execution.md) of NeMo-Run Guides. 

# In[ ]:


def local_executor_torchrun(nodes: int = 1, devices: int = 1) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

if __name__ == '__main__':
    run.run(configure_finetuning_recipe(), executor=local_executor_torchrun())

