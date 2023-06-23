#!/bin/bash

# set default
user=tran
only_train=0
only_inference=0
only_english=0
only_chinese=0
per_gpu_batch_size=8
seed=42
translate_train=0
cache_dir=.
translate_mode=""
target_langs="["fr", "vi"]"
source_lang=en
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

dev_dir=dev_$fname.json

sbatch <<EOT
#!/bin/bash
#
#SBATCH --job-name="$fname"
#SBATCH --output=/ukp-storage-1/$user/res.%j.%N."$fname".txt
#SBATCH --mail-user=an.tran@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --account=ukp-student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --constraint="gpu_model:a100|gpu_model:a6000|gpu_model:a180"
#SBATCH --gres=gpu:1

nvidia-smi
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh
conda activate acl23doc2dial
module purge
module load cuda

pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user/acl23doc2dial/ &&\
export PYTHONHASHSEED=$seed &&\
export SEED_VALUE=$seed &&\


if [[ $lang_token -eq 1 ]]
then
    python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --target-langs "$target_langs" --source-lang $source_lang --batch-accumulation --per-gpu-batch-size $per_gpu_batch_size  --cache-dir $fname --eval-input-file $dev_dir --eval-lang "$eval_lang" --translate-mode $translate_mode --lang-token
else
    python /ukp-storage-1/$user/acl23doc2dial/pipeline.py --target-langs "$target_langs" --source-lang $source_lang --batch-accumulation --per-gpu-batch-size $per_gpu_batch_size  --cache-dir $fname --eval-input-file $dev_dir --eval-lang "$eval_lang" --translate-mode $translate_mode
fi
popd
exit 0
EOT

# --eval-lang [['fr', 'vi', 'en'], ['fr', 'vi'], ['fr'], ['vi'], ['en']]