# Multilingual Grounded Dialog Models
## Project Structure
```bash
.
├── all_passages
├── args.txt                      # [deprecated] storing arguments to pass to slurm_script.sh 
├── cheatsheet.md
├── data_exploration              # data analysis & exploration scripts
│   ├── dummy                     # storing dummy data
│   └── error_anaylsis.py         # executing an (extensive) error analysis on all results
├── inference_generation.py
├── inference_rerank.py
├── inference_retrieval.py
├── modelscope                    
├── requirements.txt              
├── run.sh                        # run experiments /wo extended datasets: fr, vn
├── run_ext.sh                    # run experiments /w extended datasets: en, cn
├── slurm_script.sh               # run commit slurm job
├── slurm_script_kwargs.sh        # commit slurm job
├── train_generation.py           
├── train_rerank.py
├── train_retrieval.py
└── utils                         # aux functions for data analysis & data preprocessing
    ├── data_exploration.py
    └── preprocessing.py
```

## `acl23doc2dial` Experiments

### Start experiments

  ```shell
  Usage: slurm_script_kwargs.sh [OPTIONS]

  Options:
    --extended                  Enable usage of extended datasets: en + cn
    --lang_token                Enable usage of additional language token <lang> 
    --per_gpu_batch_size        Specify batch size that fits into given GPU VRAM
    --fname                     Specify the cache directory and dev set filename (i.e., dev_$fname.json)
  
  Example usage
  # \wo extended datasets, i.e., only the French and Vietnamese datasets
  bash slurm_script_kwargs.sh --extended 0 --lang_token 1 --fname lang_token --per_gpu_batch_size 1
  
  # \w extended datasets, i.e., adding the English and Chinese datasets
  bash slurm_script_kwargs.sh --extended 1 --lang_token 1 --fname lang_token --per_gpu_batch_size 1
  ```
