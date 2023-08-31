import train_retrieval
import train_rerank
import train_generation

import inference_retrieval
import inference_rerank
import inference_generation

import translate_train, translate_test
from tqdm import tqdm
from utils.preprocessing import get_args
import re
import logging
import os 


if __name__ == '__main__':
    kwargs = get_args()

    if not os.path.exists(kwargs["cache_dir"]):
        os.makedirs(kwargs["cache_dir"])

    pipeline_steps = {
        "train" :       {
            # "train_retrieval":      train_retrieval,
            # "train_rerank":         train_rerank, 
            "train_generation":     train_generation
            },
        "inference" :   {
            "inference_retrieval":  inference_retrieval, 
            "inference_rerank":     inference_rerank, 
            "inference_generation": inference_generation
            }
    }

    # start training pipeline
    if not kwargs['only_inference']:
        print("START TRAINING...")

        for name, train_step in pipeline_steps['train'].items():
            
            if kwargs["translate_mode"] == "train":
                step_name = re.search(r"train_(.*)", name).group(1)
                tmp = kwargs.copy()
                tmp[f"{step_name}_step"] = True
                translate_train.main(**tmp)
                print(f"finished translate-train...")

            logging.info(f"start {name} step...")
            train_step.main(**kwargs)
            print(f"finished training...")


    # start infererence pipeline
    if not kwargs['only_train']:
        print("START INFERENCE...")

        for name, inference_step in tqdm(pipeline_steps['inference'].items()):
            print(f"start {name} step...")

            if kwargs["translate_mode"] == "test":
                step_name = re.search(r"inference_(.*)", name).group(1)
                tmp[f"{step_name}_step"] = True
                print("start test mode...")
                translate_test.main(**tmp)

            inference_step.main(**kwargs)

        print(f"finished inference...")