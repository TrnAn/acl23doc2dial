# Multilingual Grounded Dialog Models
## Project Structure
```bash
.
│   cheatsheet.md                                         # cheatsheet with example commands to start experiments
│   cn_train_dataset_retrieval_generation_hn.json         # preprocesses Chinese data containing hard negatives, i.e., BM25
│   environment.yml                                       # set up env
│   en_train_dataset_retrieval_generation_hn.json         # preprocesses English data containing hard negatives, i.e., BM25
│   en_train_dataset_retrieval_generation_in_domain.json  
│   inference_domain_clf.py
│   inference_generation.py
│   inference_rerank.py
│   inference_retrieval.py
│   length_penalty_experiments.py                         # experiment on various length penalties
│   pipeline.py                                           # pipeline script to start: retrieval, rerank, generation for training and inference
│   requirements.txt                                      # set up env
│   slurm_script_kwargs_pipeline.sh                       # SLURM script to commit job
│   train_domain_clf.py                                   # train script: retrieval with an additional domain classification step
│   train_generation.py
│   train_rerank.py
│   train_retrieval.py
│   translate_test.py
│   translate_train.py
│
├───all_passages                                          # knowledge passages
│   │
│   └───lang_token                                        # knowledge passages with prepended language token
|
├───data_exploration                                      # error analysis scripts
│
├───metrics                         
│   │   q2.py                                             # script for mQ2 metric
│   └───evaluation_sheets                                 # mQ2 metric vs human assessment sheets
│
├───models                                                # contains document_grounded_dialog_domain_clf_trainer
├───modelscope                                            # modelscope models, including re2g
└───utils                                                 # aux functions
```


## `acl23doc2dial` Experiments

### Start experiments

  ```shell
  Usage: slurm_script_kwargs_pipeline.sh [OPTIONS]

  Options:
    --apply_dclf                add domain classification step
    --length_penalty            set a length penalty, defaults to 1
    --add_n_hard_negatives      add number of hard negatives per instance 
    --target_langs              set target languages for MT experiments
    --source_langs              set source languages for MT experiments
    --per_gpu_batch_size        set batch size per gpu
    --fname                     set cache dir and dev set filename
    --eval_lang                 set evaluation language(s)
    --translate_mode            set translation mode; i.e., train: translate-train, test: translate-test
    --lang_token                add language token
    
  Example usage
  - see cheatsheet.md
  ```
