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
declare -a arr=("<fr> <vn>" "<fr>" "<vn>" "<en> <cn>" "<en>" "<cn>")

pushd /ukp-storage-1/tran/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/tran//acl23doc2dial/ &&\
echo "== START TRAINING ==" 
echo "== \WO EXTENDED DATASET =="
echo "output saved to (1) cache dir: \"$fname/\" (2) dev set saved as: \"$dev_dir\""
if [[ $lang_token -eq 1 ]]
then
    echo "\w language token"
    python /ukp-storage-1/tran/acl23doc2dial/train_retrieval.py --cache-dir=$fname  --lang-token &&\
    echo "train_retrieval finished..." &&\
    python /ukp-storage-1/tran/acl23doc2dial/train_rerank.py  --cache-dir=$fname --lang-token &&\
    echo "train_rerank finished..." &&\
    python /ukp-storage-1/tran/acl23doc2dial/train_generation.py --cache-dir=$fname --batch-accumulation --per-gpu-batch-size=$per_gpu_batch_size --eval-input-file=$dev_dir --lang-token &&\
    echo "train_generation finished..." 

    for i in "${arr[@]}"
    do
        echo "== START INFERENCE on $i ==" &&\
        python /ukp-storage-1/tran/acl23doc2dial/inference_retrieval.py --cache-dir=$fname --eval-input-file=$dev_dir --lang-token --eval-lang $i &&\
        echo "inference_retrieval finished..." &&\
        python /ukp-storage-1/tran/acl23doc2dial/inference_rerank.py --cache-dir=$fname --eval-input-file=$dev_dir --lang-token --eval-lang $i &&\
        echo "inference_rerank finished..." &&\
        python /ukp-storage-1/tran/acl23doc2dial/inference_generation.py --cache-dir=$fname --eval-input-file=$dev_dir --lang-token --eval-lang $i &&\
        echo "inference_generation finished..."
    done
else
    echo "\wo language token"
    python /ukp-storage-1/tran/acl23doc2dial/train_retrieval.py --cache-dir=$fname &&\
    echo "train_retrieval finished..." &&\
    python /ukp-storage-1/tran/acl23doc2dial/train_rerank.py  --cache-dir=$fname &&\
    echo "train_rerank finished..." &&\
    python /ukp-storage-1/tran/acl23doc2dial/train_generation.py --cache-dir=$fname --batch-accumulation --per-gpu-batch-size=$per_gpu_batch_size --eval-input-file=$dev_dir &&\
    echo "train_generation finished..."

    for i in "${arr[@]}"
    do
        echo "== START INFERENCE ON $i" &&\
        python /ukp-storage-1/tran/acl23doc2dial/inference_retrieval.py --cache-dir=$fname --eval-input-file=$dev_dir --eval-lang $i &&\
        echo "inference_retrieval finished..." &&\
        python /ukp-storage-1/tran/acl23doc2dial/inference_rerank.py --cache-dir=$fname --eval-input-file=$dev_dir --eval-lang $i &&\
        echo "inference_rerank finished..." &&\
        python /ukp-storage-1/tran/acl23doc2dial/inference_generation.py --cache-dir=$fname --eval-input-file=$dev_dir --eval-lang $i &&\
        echo "inference_generation finished..."
    done
fi
popd