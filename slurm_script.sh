#!/bin/bash
#
#SBATCH --job-name=acl23doc2dial_lang_token
#SBATCH --output=/ukp-storage-1/tran/res.%j.%N.lang_token.txt
#SBATCH --mail-user=an.tran@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --account=ukp-student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/tran/miniconda3/etc/profile.d/conda.sh
conda activate acl23doc2dial
module purge
module load cuda
bash /ukp-storage-1/tran/DAMO-ConvAI/acl23doc2dial/run.sh
