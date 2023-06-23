# Le Cheatsheet
- shared folder
    - > /storage/ukp/shared/shared_daheim_tran 
- set gpu vram:
    - `#SBATCH --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180|gpu_model:titanrtx"`
---

## Run as `SBATCH`
```bash
# running train + inference /wo extended dataset en+cn
bash slurm_script_kwargs.sh --extended 0 --lang_token 1 --fname lang_token --per_gpu_batch_size 1 
# running train + inference /w extended dataset en+cn
bash slurm_script_kwargs.sh --extended 1 --lang_token 1 --fname ext_lang_token --per_gpu_batch_size 1 
# only running inference
bash slurm_script_kwargs.sh --extended 0 --lang_token 1 --fname lang_token --per_gpu_batch_size 1 --only_inference 1
# only running train
bash slurm_script_kwargs.sh --extended 1 --lang_token 1 --fname en_lang_token --only_train 1 --only_english 1

# running experiments on both english & chinese datasets
bash slurm_script_kwargs.sh --extended 1 --lang_token 1 --fname ext_lang_token --per_gpu_batch_size 1

# running experiments on english dataset
bash slurm_script_kwargs.sh --extended 1 --lang_token 1 --fname en_ext_lang_token --per_gpu_batch_size 1 --only_english 1
bash slurm_script_kwargs.sh --extended 1 --lang_token 0 --fname en_no_lang_token  --only_english 1

# running experiments on chinese dataset
bash slurm_script_kwargs.sh --extended 1 --lang_token 1 --fname cn_ext_lang_token --per_gpu_batch_size 1 --only_chinese 1
bash slurm_script_kwargs.sh --extended 1 --lang_token 0 --fname cn_ext_no_lang_token --per_gpu_batch_size 1 --only_chinese 1
```

## Start `SRUN` 
```bash
srun --qos yolo -p yolo --gres=gpu:1 --mem-per-cpu=24g --pty bash
srun --qos yolo -p yolo --gres=gpu:1 --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180|gpu_model:titanrtx" --mem-per-cpu=42g --pty bash
srun -p testing --gres=gpu:1 --mem-per-cpu=24g --pty bash
```

## Run as `SRUN` 

```bash
user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
mode=train &&\
seed=42 &&\
fname=ttrain_lang_token &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --translate-mode $mode --batch-accumulation --per-gpu-batch-size 8 --cache-dir=$fname --lang-token --eval-input-file=$dev_dir --eval-lang "[['fr', 'vi', 'en'], ['fr', 'vi'], ['fr'], ['vi'], ['en']]" --target-langs "['fr', 'vi']" --source-lang "en" &&\
echo "pipeline finished..." &&\
popd
```
```sh
bash slurm_script_kwargs_pipeline.sh --fname ttrain_lang_token --translate_mode train --lang_token 1 --per_gpu_batch_size 4 --eval_lang "[['fr', 'vi', 'en'], ['fr', 'vi'], ['fr'], ['vi'], ['en']]" --target_langs "['fr', 'vi']" --source_lang "en"
```

```bash
seed=42 &&\
user=tran &&\
only_chinese=0 &&\
only_english=1 &&\
fname=en_lang_token &&\
dev_dir=$fname\/dev_$fname.json &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTHONHASHSEED=$seed &&\
export SEED_VALUE=$seed&&\
python /ukp-storage-1/$user/acl23doc2dial/train_rerank.py  --extended-dataset --cache-dir=$fname --lang-token --only-english=1&&\
echo "train_rerank finished..." &&\
popd
````