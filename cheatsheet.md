# Le Cheatsheet
- shared folder
    - > /storage/ukp/shared/shared_daheim_tran 
- set gpu vram:
    - `#SBATCH --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180|gpu_model:titanrtx"`
---

## Run as `SBATCH`
```bash

```

## Start `SRUN` 
```bash
srun --qos yolo -p yolo --gres=gpu:1 --mem-per-cpu=24g --pty bash
srun --qos gpu-athene --gres=gpu:1 --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180|gpu_model:titanrtx" --mem-per-cpu=42g --pty bash 
srun -p testing --gres=gpu:1 --mem-per-cpu=24g --pty bash
```

## Run as `SRUN` 

```bash
user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
mode=train &&\
seed=42 &&\
fname=tst_pipeline &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --translate-mode $mode --batch-accumulation --per-gpu-batch-size 8 --cache-dir=$fname --lang-token --eval-input-file=$dev_dir --eval-lang "[['fr', 'vi', 'en'], ['fr', 'vi'], ['fr'], ['vi'], ['en']]" --target-langs "['fr', 'vi']" --source-lang "en" &&\
echo "pipeline finished..." &&\
popd


user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
mode=domain_clf &&\
seed=42 &&\
fname=tst_pipeline &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
python /ukp-storage-1/$user/acl23doc2dial/train_domain_clf.py --batch-accumulation --per-gpu-batch-size 32 --cache-dir=$fname --eval-input-file=$dev_dir --eval-lang "[['fr', 'vi'], ['fr'], ['vi']]" --target-langs "[]" --source-lang "[]" &&\
popd
```

```sh
bash slurm_script_kwargs_pipeline.sh --fname ttrain_hn --translate_mode train --lang_token 1 --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi', 'en'], ['fr', 'vi'], ['fr'], ['vi'], ['en']]" --target_langs "['fr', 'vi']" --source_langs "['en']"

bash slurm_script_kwargs_pipeline.sh --fname ttest_improved --translate_mode test --lang_token 1 --per_gpu_batch_size 1 --eval_lang "[['fr','vi'], ['fr'], ['vi']]" --target_langs "['en']" --source_langs "['fr', 'vi']" --source_langs "['en']"

bash slurm_script_kwargs_pipeline.sh --fname translate_test_lang_token_lp --translate_mode test --lang_token 1 --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi'], ['fr'], ['vi']]" --target_langs "['en']" --source_langs "['fr', 'vi']"

bash slurm_script_kwargs_pipeline.sh --fname all_lang_token --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['fr','vi', 'en'], ['fr','vi'], ['fr'], ['vi'], ['en']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh --fname cn_vi_lang_token --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['cn','vi'], ['vi']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh --fname en_fr_lang_token --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['fr','en'], ['fr']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh --equal_dataset_size 1 --fname en_no_lang_token_eq --lang_token 1 --per_gpu_batch_size 1 --eval_lang "[['fr','en'], ['fr']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh --equal_dataset_size 1 --fname fr_vi_en_no_lang --lang_token 0 --per_gpu_batch_size 1 --eval_lang "[['fr','vi','en'], ['fr','vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh  --fname fr_vi_no_lang --lang_token 0 --per_gpu_batch_size 8 --eval_lang "[['fr','vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh --fname retrieval50 --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['fr','vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh --fname domain_clf_lang_token  --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['fr','vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]"

bash slurm_script_kwargs_pipeline.sh --fname fr_lang --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['fr']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh --fname hard_negatives_no_fn --add_n_hard_negatives 1 --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['fr','vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]"
bash slurm_script_kwargs_pipeline.sh --fname hard_negatives2 --add_n_hard_negatives 2 --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['fr','vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]"

bash slurm_script_kwargs_pipeline.sh --fname hard_negatives8 --add_n_hard_negatives 8 --lang_token 1 --per_gpu_batch_size 8 --eval_lang "[['fr','vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]"
```


```bash
seed=42 &&\
user=tran &&\
only_chinese=0 &&\
only_english=1 &&\
fname=ttest_test &&\
dev_dir=dev_$fname.json &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTHONHASHSEED=$seed &&\
export SEED_VALUE=$seed&&\
python /ukp-storage-1/$user/acl23doc2dial/pipeline.py  --lang-token --translate-mode test --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --eval-input-file $dev_dir --eval-lang "[['fr','vi'], ['fr'], ['vi']]" --target-langs "['en']" --source-langs "['fr', 'vi']" &&\
echo "train_rerank finished..." &&\
popd
````

user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
seed=42 &&\
fname=ttrain_tokens &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  &&\
python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --lang-token --translate-mode train --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --eval-input-file $dev_dir --eval-lang "[['fr','vi'], ['fr'], ['vi']]" --target-langs "['fr','vi']" --source-langs "['en']" &&\
echo "pipeline finished..." &&\
popd


user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
seed=42 &&\
fname=tst_negative_duplicates &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  &&\
python /ukp-storage-1/$user/acl23doc2dial/train_retrieval.py --n-hard-negatives 1 --lang-token --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --eval-input-file $dev_dir --eval-lang "[['fr','vi'], ['fr'], ['vi']]" --target-langs "[]" --source-langs "[]" &&\
echo "pipeline finished..." &&\
popd


user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
seed=42 &&\
fname=translate_test_lang_token_lp &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  &&\
python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --lang-token --translate-mode test --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --eval-input-file $dev_dir --eval-lang "[['fr','vi'], ['fr'], ['vi']]" --target-langs "['en']" --source-langs "['fr','vi']" &&\
echo "pipeline finished..." &&\
popd


user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
mode=test &&\
seed=42 &&\
fname=all_lang_token &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  &&\
python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --lang-token --eval-input-file $dev_dir --eval-lang "[['fr','vi', 'en'], ['fr','vi'], ['fr'], ['vi'], ['en']]" --target-langs "[]" --source-langs "[]" &&\
echo "pipeline finished..." &&\
popd


user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
seed=42 &&\
fname=domain_clf &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
python /ukp-storage-1/$user/acl23doc2dial/inference_rerank.py --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --eval-input-file $dev_dir --eval-lang "[['fr','vi'], ['fr'], ['vi']]" --target-langs "[]" --source-langs "[]" &&\
echo "pipeline finished..." &&\
popd

user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
seed=42 &&\
fname=fr_vi_no_lang &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
python /ukp-storage-1/$user/acl23doc2dial/inference_rerank.py --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --eval-input-file $dev_dir --eval-lang "[['fr','vi']]" --target-langs "[]" --source-langs "[]" &&\
echo "pipeline finished..." &&\
popd



user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
mode=test &&\
seed=42 &&\
fname=en_fr_lang_token_eq &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 &&\
python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --equal-dataset-size 1 --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --lang-token --eval-input-file $dev_dir --eval-lang "[['en','fr'], ['fr']]" --target-langs "[]" --source-langs "[]" > "$fname".out.txt &&\
echo "pipeline finished..." &&\
popd


user=tran &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\ 
mode=test &&\
seed=42 &&\
fname=cn_vi_lang_token_eq &&\
dev_dir=dev_$fname.json &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 &&\
python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --equal-dataset-size 1 --batch-accumulation --per-gpu-batch-size 1 --cache-dir $fname --lang-token --eval-input-file $dev_dir --eval-lang "[['vi','cn'], ['cn'], ['vi']]" --target-langs "[]" --source-langs "[]" > "$fname".out.txt &&\
echo "pipeline finished..." &&\
popd
