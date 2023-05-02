#!/bin/bash
#
pushd /ukp-storage-1/tran/acl23doc2dial/ &&\
export HOME=/ukp-storage-1/tran//acl23doc2dial/ &&\
echo "== START TRAINING ==" &&\
python /ukp-storage-1/tran/acl23doc2dial/train_retrieval.py  @args.txt &&\
echo "train_retrieval finished..." &&\
python /ukp-storage-1/tran/acl23doc2dial/train_rerank.py  @args.txt &&\
echo "train_rerank finished..." &&\
python /ukp-storage-1/tran/acl23doc2dial/train_generation.py @args.txt &&\
echo "train_generation finished..." &&\
popd