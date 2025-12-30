export NEMORUN_HOME := /root/work/run/nm2511-perf
NSTEP := 25

# pretrain.1. 8xB200, BF16
llama3-bf16:
	python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $(NSTEP)

# pretrain.2. 8xB200, FP8 (delayed scaling)
llama3-fp8-ds:
	python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $(NSTEP) \
		--compute_dtype fp8 --fp8_recipe ds

# pretrain.3. 8xB200, MXFP8
llama3-mxfp8:
	python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $(NSTEP) \
		--compute_dtype fp8 --fp8_recipe mxfp8

# require 
# pretrain.4. 8xB200, NVFP4
llama3-nvfp4:
	python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $(NSTEP) \
		--compute_dtype fp8 --fp8_recipe nvfp4

# Qwen3 30B A3B  8xB200, MXFP8
qwen3-30b-a3b-mxfp8:
	python -m scripts.performance.llm.pretrain_qwen3_30b_a3b -g b200 --max_steps $(NSTEP) \
		--compute_dtype fp8 --fp8_recipe mxfp8

# Qwen3 30B A3B  8xB200, NVFP4
qwen3-30b-a3b-nvfp4:
	python -m scripts.performance.llm.pretrain_qwen3_30b_a3b -g b200 --max_steps $(NSTEP) \
		--compute_dtype fp8 --fp8_recipe nvfp4