#!/bin/bash

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
#SBATCH --output=/ukp-storage-1/tran/res.%j.%N."$fname".txt
#SBATCH --mail-user=an.tran@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --account=ukp-student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:v100:1

nvidia-smi
source /ukp-storage-1/tran/miniconda3/etc/profile.d/conda.sh
conda activate acl23doc2dial
module purge
module load cuda

if [[ $extended -eq 1 ]]
then
    echo "run script /w extended datasets..."
    bash /ukp-storage-1/tran/acl23doc2dial/run_ext.sh --lang_token $lang_token --fname $fname --per_gpu_batch_size $per_gpu_batch_size
else
    echo "run script /wo extended datasets..."
    bash /ukp-storage-1/tran/acl23doc2dial/run.sh --lang_token $lang_token --fname $fname --per_gpu_batch_size $per_gpu_batch_size
fi

exit 0
EOT

