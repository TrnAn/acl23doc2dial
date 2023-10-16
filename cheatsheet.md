# Le Cheatsheet
- set gpu vram:
    - `#SBATCH --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180|gpu_model:titanrtx"`
---

## Start `SRUN` 
```bash
srun --qos yolo -p yolo --gres=gpu:1 --mem-per-cpu=24g --pty bash
srun --qos gpu-athene --gres=gpu:1 --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180" --mem-per-cpu=42g --pty bash 
srun -p testing --gres=gpu:1 --mem-per-cpu=24g --pty bash
```

## Run as `SBATCH` 

```sh
# experiment baseline using: fr, vi
bash slurm_script_kwargs_pipeline.sh --lang_token 0 --fname 0_baseline --batch_accumulation --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]" --length_penalty 1

# finetune on languages separately
bash slurm_script_kwargs_pipeline.sh --lang_token 0 --fname 0_baseline_vi --batch_accumulation --per_gpu_batch_size 1 --eval_lang "[['vi']]" --target_langs "[]" --source_langs "[]" --length_penalty 1
bash slurm_script_kwargs_pipeline.sh --lang_token 0 --fname 0_baseline_fr --batch_accumulation --per_gpu_batch_size 1 --eval_lang "[['fr']]" --target_langs "[]" --source_langs "[]" --length_penalty 1

# start language token experiment
bash slurm_script_kwargs_pipeline.sh --lang_token 1 --fname 1_baseline_lt_old --batch_accumulation --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]" --length_penalty 1

# start translate-train experiment: use MT data for finetuning
bash slurm_script_kwargs_pipeline.sh --fname ttrain_2epochs_nlt --translate_mode train --lang_token 0 --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi', 'en'], ['fr', 'vi'], ['fr'], ['vi']]" --target_langs "['fr', 'vi']" --source_langs "['en']" --length_penalty 1

# continue to train translate-train wo/ MT data
bash slurm_script_kwargs_pipeline.sh --fname ttrain_2epochs_nlt --lang_token 0 --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi'], ['fr'], ['vi']]" --target_langs "['fr', 'vi']" --source_langs "['en']" --length_penalty 1

# start translate-test experiment
bash slurm_script_kwargs_pipeline.sh --fname ttest_lt --translate_mode test --lang_token 1 --per_gpu_batch_size 1 --eval_lang "[['fr','vi'], ['fr'], ['vi']]" --target_langs "['en']" --source_langs "['fr', 'vi']" --length_penalty 1

# start domain clf experiment
bash slurm_script_kwargs_pipeline.sh --apply_dclf 1 --lang_token 0 --fname domain_clf_10epochs --batch_accumulation --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]" --length_penalty 1

# IN-BATCH NEGATIVES EXPERIMENT
bash slurm_script_kwargs_pipeline.sh --lang_token 0 --fname 0_baseline_2hn --batch_accumulation --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]" --length_penalty 1 --add_n_hard_negatives 1
bash slurm_script_kwargs_pipeline.sh --lang_token 0 --fname 0_baseline_4hn --batch_accumulation --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]" --length_penalty 1 --add_n_hard_negatives 3
bash slurm_script_kwargs_pipeline.sh --lang_token 0 --fname 0_baseline_6hn --add_n_hard_negatives 5 --batch_accumulation  --per_gpu_batch_size 1 --eval_lang "[['fr', 'vi'], ['fr'], ['vi']]" --target_langs "[]" --source_langs "[]" --length_penalty 1 

```

## Start `SRUN` Experiments
### $mQ^2$ metric experiment on baseline
```bash
# for fr+vi
seed=42 &&\
user=tran &&\
fname=0_baseline &&\
dev_dir=dev_$fname.json &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTHONHASHSEED=$seed &&\
export SEED_VALUE=$seed&&\
mkdir -p ./$fname/ &&\
script -q -c "python /ukp-storage-1/'$user'/acl23doc2dial/metrics/q2.py" | tee -a ./$fname/log_output.txt &&\
echo "train_rerank finished..." &&\
popd

# for French subset
user=tran &&\
fname=0_baseline &&\
dev_dir=dev_$fname.json &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTHONHASHSEED=$seed &&\
export SEED_VALUE=$seed&&\
mkdir -p ./$fname/ &&\
script -q -c "python /ukp-storage-1/'$user'/acl23doc2dial/metrics/q2.py --eval-lang \"['fr']\" --sample-size 50" | tee -a ./$fname/log_output.txt &&\
popd


# for Vietnamese subset
user=tran &&\
fname=0_baseline &&\
dev_dir=dev_$fname.json &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTHONHASHSEED=$seed &&\
export SEED_VALUE=$seed&&\
mkdir -p ./$fname/ &&\
script -q -c "python /ukp-storage-1/'$user'/acl23doc2dial/metrics/q2.py --eval-lang \"['vi']\" --sample-size 50" | tee -a ./$fname/log_output.txt &&\
popd

```

### Length Penalty Experiment
```bash
seed=42 &&\
user=tran &&\
fname=0_baseline &&\
dev_dir=dev_$fname.json &&\
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh &&\
conda activate acl23doc2dial &&\
pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTHONHASHSEED=$seed &&\
export SEED_VALUE=$seed&&\
mkdir -p ./$fname/ &&\
script -q -c "python /ukp-storage-1/'$user'/acl23doc2dial/length_penalty_experiments.py" | tee -a ./$fname/log_output_lp.txt &&\
echo "lp experiments finished..." &&\
popd
```