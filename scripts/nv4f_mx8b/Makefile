
# WIP: somehow nemo experiment not seeing the env vars but interative execution can

# Important: do logging HF and Wandb

export NEMO_HOME=/root/work/nemo # where HF/NeMo/stored
export NEMORUN_HOME=/root/work/run/eval-dev
export OPENSCIENCE_DS=/root/work/run/qat_experiment/openscience_proc #preprocessed openscience dataset, piggyback trtmo qat runs
export gpu_type=b200 # we assume 8x
export wandb_project=tut-nemo-sft
# export wandb_project = faster-qat


f8ds-sft-qwen3-8b:
	python -m scripts.nv4f_mx8b.finetune_qwen3_8b \
		-g $(gpu_type) -gn 1 -ng 8 \
		-f sft \
		-wdp $(wandb_project) \
		-dp $(OPENSCIENCE_DS) \
		-mb 1 -gb 512 -tp 1 -pp 1 -cp 1 -vp 1 -ep 1 -et 1 \
		--compute_dtype fp8 --fp8_recipe ds

mxf8-sft-qwen3-8b:
	python -m scripts.nv4f_mx8b.finetune_qwen3_8b \
		-g $(gpu_type) -gn 1 -ng 8 \
		-f sft \
		-wdp $(wandb_project) \
		-dp $(OPENSCIENCE_DS) \
		-mb 1 -gb 512 -tp 1 -pp 1 -cp 1 -vp 1 -ep 1 -et 1 \
		--compute_dtype fp8 --fp8_recipe mxfp8

nv4f_mx8b-sft-qwen3-8b:
	python -m scripts.nv4f_mx8b.finetune_qwen3_8b \
		-g $(gpu_type) -gn 1 -ng 8 \
		-f sft \
		-wdp $(wandb_project) \
		-dp $(OPENSCIENCE_DS) \
		-mb 1 -gb 512 -tp 1 -pp 1 -cp 1 -vp 1 -ep 1 -et 1 \
		--compute_dtype fp8 --fp8_recipe nv4f_mx8b

