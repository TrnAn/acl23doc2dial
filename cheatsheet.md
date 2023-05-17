# Le Cheatsheet
- shared folder
    - > /storage/ukp/shared/shared_daheim_tran 
- set gpu vram:
    - `#SBATCH --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180|gpu_model:titanrtx"`
---

## Run as `SBATCH`
```bash
bash slurm_script_kwargs.sh --extended 1 --lang_token 1 --fname ext_lang_token --per_gpu_batch_size 4
```

## Start `SRUN` 
```bash
srun --qos yolo -p yolo --gres=gpu:1 --mem-per-cpu=24g --pty bash
srun -p testing --gres=gpu:1 --mem-per-cpu=24g --pty bash
```

## Run as `SRUN` 

```bash
pushd /ukp-storage-1/tran/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/tran//acl23doc2dial/ &&\
fname=ext_no_lang_token &&\
dev_dir=$fname\/dev_$fname.json &&\
python /ukp-storage-1/tran/acl23doc2dial/train_generation.py --cache-dir=$fname --extended-dataset --batch-accumulation --per-gpu-batch-size=1 --eval-input-file=$dev_dir &&\
popd
```