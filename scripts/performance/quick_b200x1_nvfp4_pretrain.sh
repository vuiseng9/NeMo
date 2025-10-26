export CUBLASLT_LOG_LEVEL=2
export CUBLASLT_LOG_FILE=./log.cublaslt

export NEMORUN_HOME=./quickrun-nvfp4
NSTEP=10
python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $NSTEP --compute_dtype fp8 --fp8_recipe nvfp4 \
    -ng 1 -gn 1 -mb 1 -gb 32 -tp 1 -pp 1 -cp 1 -vp 1 -ep 1 -et 1 