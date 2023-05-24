#!/bin/bash

# set default
user=tran
only_train=0
only_inference=0
only_english=0
only_chinese=0


while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

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
#SBATCH --constraint="gpu_model:a100|gpu_model:v100|gpu_model:a6000|gpu_model:a180"
#SBATCH --gres=gpu:1

nvidia-smi
source /ukp-storage-1/$user/miniconda3/etc/profile.d/conda.sh
conda activate acl23doc2dial
module purge
module load cuda

pushd /ukp-storage-1/$user/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/$user//acl23doc2dial/ &&\
if [[ $extended -eq 1 ]]
then
    echo "run script /w extended datasets..."
    bash /ukp-storage-1/$user/acl23doc2dial/run_ext.sh --user $user --lang_token $lang_token --fname $fname --per_gpu_batch_size $per_gpu_batch_size --only_train $only_train --only_inference $only_inference --only_english $only_english --only_chinese $only_chinese
else
    echo "run script /wo extended datasets..."
    bash /ukp-storage-1/$user/acl23doc2dial/run.sh --user $user --lang_token $lang_token --fname $fname --per_gpu_batch_size $per_gpu_batch_size --only_train $only_train --only_inference $only_inference
fi
popd
exit 0
EOT

