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
# pushd /ukp-storage-1/tran/acl23doc2dial/ &&\
# export HOME=/ukp-storage-1/tran//acl23doc2dial/ &&\
echo "== START TRAINING ==" &&\
echo "== \W EXTENDED DATASET =="
echo "output saved to (1) cache dir: \"$fname/\" (2) dev set saved as: \"$dev_dir\" (3) inference only: $only_inference (4) train_only: $only_train"

if [[ $only_inference -eq 0 ]]
then
    if [[ $lang_token -eq 1 ]]
    then
        echo "started train \w language token" &&\
        python /ukp-storage-1/tran/acl23doc2dial/train_retrieval.py --extended-dataset --batch-accumulation --cache-dir=$fname --lang-token --eval-input-file=$dev_dir --only-chinese=$only_chinese --only-english=$only_english &&\
        echo "train_retrieval finished..." &&\
        python /ukp-storage-1/tran/acl23doc2dial/train_rerank.py  --extended-dataset --cache-dir=$fname --lang-token &&\
        echo "train_rerank finished..." &&\
        python /ukp-storage-1/tran/acl23doc2dial/train_generation.py --extended-dataset --cache-dir=$fname --batch-accumulation --per-gpu-batch-size=$per_gpu_batch_size --eval-input-file=$dev_dir --lang-token --only-chinese=$only_chinese --only-english=$only_english&&\
        echo "train_generation finished..." 
    else
        echo "started train \wo language token" &&\
        python /ukp-storage-1/tran/acl23doc2dial/train_retrieval.py --extended-dataset --batch-accumulation --cache-dir=$fname --eval-input-file=$dev_dir --only-chinese=$only_chinese --only-english=$only_english &&\
        echo "train_retrieval finished..." &&\
        python /ukp-storage-1/tran/acl23doc2dial/train_rerank.py  --extended-dataset --cache-dir=$fname &&\
        echo "train_rerank finished..." &&\
        python /ukp-storage-1/tran/acl23doc2dial/train_generation.py --extended-dataset --cache-dir=$fname --batch-accumulation --per-gpu-batch-size=$per_gpu_batch_size --eval-input-file=$dev_dir --only-chinese=$only_chinese --only-english=$only_english&&\
        echo "train_generation finished..." 
    fi
fi
echo "== START INFERENCE =="
if [[ $only_english -eq 1 ]]
then
    declare -a arr=("fr vi" "fr" "vi" "en" )
fi

if [[ $only_chinese -eq 1 ]]
then
    declare -a arr=("fr vi" "fr" "vi" "cn")
fi

if [[ $only_english -eq 0 && $only_chinese -eq 0 ]]
then
    declare -a arr=("fr vi" "fr" "vi" "en cn" "en" "cn")
fi

if [[ $only_inference -eq 0 ]]
then
    for i in "${arr[@]}"
    do
        echo "== START INFERENCE on $i =="
        if [[ $lang_token -eq 1 ]]
        then
                python /ukp-storage-1/tran/acl23doc2dial/inference_retrieval.py --extended-dataset --cache-dir=$fname --eval-input-file=$dev_dir --lang-token --eval-lang $i --only-chinese=$only_chinese --only-english=$only_english &&\
                echo "inference_retrieval finished..." &&\
                python /ukp-storage-1/tran/acl23doc2dial/inference_rerank.py --extended-dataset --cache-dir=$fname --eval-input-file=$dev_dir --lang-token --eval-lang $i --only-chinese=$only_chinese --only-english=$only_english &&\
                echo "inference_rerank finished..." &&\
                python /ukp-storage-1/tran/acl23doc2dial/inference_generation.py --extended-dataset --cache-dir=$fname --eval-input-file=$dev_dir --lang-token --eval-lang $i --only-chinese=$only_chinese --only-english=$only_english
        else
                python /ukp-storage-1/tran/acl23doc2dial/inference_retrieval.py --extended-dataset --cache-dir=$fname --eval-input-file=$dev_dir --eval-lang $i --only-chinese=$only_chinese --only-english=$only_english &&\
                echo "inference_retrieval finished..." &&\
                python /ukp-storage-1/tran/acl23doc2dial/inference_rerank.py --extended-dataset --cache-dir=$fname --eval-input-file=$dev_dir --eval-lang $i --only-chinese=$only_chinese --only-english=$only_english &&\
                echo "inference_rerank finished..." &&\
                python /ukp-storage-1/tran/acl23doc2dial/inference_generation.py --extended-dataset --cache-dir=$fname --eval-input-file=$dev_dir --eval-lang $i --only-chinese=$only_chinese --only-english=$only_english
        fi        
    done
echo "inference_generation finished..."
fi
# popd