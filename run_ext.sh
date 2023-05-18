#!/bin/bash
#
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

dev_dir=$fname\/dev_$fname.json
pushd /ukp-storage-1/tran/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/tran//acl23doc2dial/ &&\
echo "== START TRAINING ==" &&\
echo "== \W EXTENDED DATASET =="
echo "output saved to (1) cache dir: \"$fname/\" (2) dev set saved as: \"$dev_dir\""

if [[ $lang_token -eq 1 ]]
then
    echo "started train_generation \w language token" &&\
    python /ukp-storage-1/tran/acl23doc2dial/train_generation.py --cache-dir=$fname --extended-dataset --batch-accumulation --per-gpu-batch-size=$per_gpu_batch_size --eval-input-file=$dev_dir --lang-token
else
    echo "started train_generation \wo language token" &&\
    python /ukp-storage-1/tran/acl23doc2dial/train_generation.py --cache-dir=$fname --extended-dataset --batch-accumulation --per-gpu-batch-size=$per_gpu_batch_size --eval-input-file=$dev_dir 
fi
echo "train_generation finished..." &&\

echo "== START INFERENCE ==" &&\
declare -a arr=("<fr> <vn>" "<fr>" "<vn>" "<en> <cn>" "<en>" "<cn>")

for i in "${arr[@]}"
do
   echo "== START INFERENCE ON $i"

   if [[ $lang_token -eq 1 ]]
   then
        python /ukp-storage-1/tran/acl23doc2dial/inference_generation.py --cache-dir=$fname --extended-dataset --eval-input-file=$dev_dir --lang-token --eval-lang $i
   else
        python /ukp-storage-1/tran/acl23doc2dial/inference_generation.py --cache-dir=$fname --extended-dataset --eval-input-file=$dev_dir --eval-lang $i
   fi        
done

echo "inference_generation finished..." &&\
popd