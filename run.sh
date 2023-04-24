#!/bin/bash
#
pushd /ukp-storage-1/tran/DAMO-ConvAI/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/tran/DAMO-ConvAI/acl23doc2dial/ &&\
python /ukp-storage-1/tran/DAMO-ConvAI/acl23doc2dial/train_retrieval.py &&\
echo "train_retrieval finished..." &&\
python /ukp-storage-1/tran/DAMO-ConvAI/acl23doc2dial/train_rerank.py &&\
echo "train_rerank finished..." &&\
python /ukp-storage-1/tran/DAMO-ConvAI/acl23doc2dial/train_generation.py &&\
echo "train_generation finished..." &&\
popd