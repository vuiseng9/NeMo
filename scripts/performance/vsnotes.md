python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps 100

8xb200-dc       Wed Oct  1 03:37:50 2025  580.82.07
[0] NVIDIA B200 | 67°C, 100 % | 124906 / 183359 MB | root(124892M)
[1] NVIDIA B200 | 56°C, 100 % | 125140 / 183359 MB | root(125126M)
[2] NVIDIA B200 | 60°C, 100 % | 125170 / 183359 MB | root(125156M)
[3] NVIDIA B200 | 68°C, 100 % | 125122 / 183359 MB | root(125108M)
[4] NVIDIA B200 | 59°C, 100 % | 125170 / 183359 MB | root(125156M)
[5] NVIDIA B200 | 66°C,  95 % | 125170 / 183359 MB | root(125156M)
[6] NVIDIA B200 | 60°C,  95 % | 125170 / 183359 MB | root(125156M)
[7] NVIDIA B200 | 68°C, 100 % | 125170 / 183359 MB | root(125156M)

Training epoch 0, iteration 45/99 | lr: 6.897e-06 | global_batch_size: 128 | global_step: 45 | max_memory_reserved: 127785762816 | max_memory_allocated: 115792977920 | reduced_train_loss: 11.96 | train_step_timing in s: 6.08 | TFLOPS_per_GPU: 1.249e+03 | consumed_samples: 5888
Training epoch 0, iteration 46/99 | lr: 7.046e-06 | global_batch_size: 128 | global_step: 46 | max_memory_reserved: 127785762816 | max_memory_allocated: 115792977920 | reduced_train_loss: 11.96 | train_step_timing in s: 6.061 | TFLOPS_per_GPU: 1.252e+03 | consumed_samples: 6016
Training epoch 0, iteration 47/99 | lr: 7.196e-06 | global_batch_size: 128 | global_step: 47 | max_memory_reserved: 127785762816 | max_memory_allocated: 115792977920 | reduced_train_loss: 11.96 | train_step_timing in s: 6.082 | TFLOPS_per_GPU: 1.248e+03 | consumed_samples: 6144
Training epoch 0, iteration 48/99 | lr: 7.346e-06 | global_batch_size: 128 | global_step: 48 | max_memory_reserved: 127785762816 | max_memory_allocated: 115792977920 | reduced_train_loss: 11.96 | train_step_timing in s: 6.071 | TFLOPS_per_GPU: 1.25e+03 | consumed_samples: 6272
Training epoch 0, iteration 49/99 | lr: 7.496e-06 | global_batch_size: 128 | global_step: 49 | max_memory_reserved: 127785762816 | max_memory_allocated: 115792977920 | reduced_train_loss: 11.96 | train_step_timing in s: 6.06 | TFLOPS_per_GPU: 1.253e+03 | consumed_samples: 6400

  global_batch_size: 128
  micro_batch_size: 2
  num_train_samples: 12800
  seq_length: 8192

  
  Tokens/sec (global) = (GBS × L) / t_step = 1,048,576 / 6.06 = 173032
  Tokens/sec/GPU = Tokens/sec (global) / 8 = 173032 / 8 = 21629

pretrain_llama3_8b_bf16_1nodes_tp1_pp1_cp1_vp1_2mbs_128gbs_1759289474


export NEMORUN_HOME=/opt/NeMo/scripts/performance/perf-run
python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps 50 --compute_dtype fp8 --fp8_recipe ds 

Tokens per step (global) = GBS × L = 128 × 8192 = 1,048,576
Tokens/sec (global) = (GBS × L) / t_step = 1,048,576 / 4.15 = 252669
Tokens/sec/GPU = Tokens/sec (global) / 8 = 252669 / 8 = 31583

export NEMORUN_HOME=/opt/NeMo/scripts/performance/perf-run
python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps 50 --compute_dtype fp8 --fp8_recipe mxfp8 