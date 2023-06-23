import train_retrieval
import train_rerank
import train_generation

import inference_retrieval
import inference_rerank
import inference_generation

import translate
from tqdm import tqdm
from utils.preprocessing import get_args
import re
import logging
import os 


if __name__ == '__main__':
    kwargs = get_args()

    if not os.path.exists(kwargs["cache_dir"]):
        os.makedirs(kwargs["cache_dir"])

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers= [
            logging.FileHandler(f'{kwargs["cache_dir"]}/pipeline.log'),  # File handler
            logging.StreamHandler()  # Console handler
        ]
    )
    logger = logging.getLogger()

    pipeline_steps = {
        "train" :       {
            "train_retrieval":      train_retrieval,
            "train_rerank":         train_rerank, 
            "train_generation":     train_generation
            },
        "inference" :   {
            "inference_retrieval":  inference_retrieval, 
            "inference_rerank":     inference_rerank, 
            }
    }

    # start training pipeline
    if not kwargs['only_inference']:
        logger.info("START TRAINING...")

        for name, train_step in pipeline_steps['train'].items():
            
            if kwargs["translate_mode"] == "train":
                step_name = re.search(r"train_(.*)", name).group(1)
                tmp = kwargs.copy()
                tmp[f"{step_name}_step"] = True
                translate.main(**tmp)
                logger.info(f"finished translate-train...")

            logging.info(f"start {name} step...")
            train_step.main(**kwargs)
            logger.info(f"finished training...")


    # start infererence pipeline
    if not kwargs['only_train']:
        unique_langs = set(item for sublist in kwargs["eval_lang"] for item in sublist)

        logger.info("START INFERENCE...")

        for name, inference_step in tqdm(pipeline_steps['inference'].items()):
            logger.info(f"start {name} step...")
                
            for idx, lang in enumerate(kwargs['eval_lang']):
                tmp = kwargs.copy()
                tmp["save_output"]  = 1 if idx == 0 else 0
                tmp["eval_lang"]    = [lang]

                if kwargs["translate_mode"] == "test":
                    for source_lang in unique_langs:
                        tmp["source_lang"] = source_lang
                        tmp["target_langs"] = ['en']
                        translate.main(**tmp)

                inference_step.main(**tmp)

        # infererence generation runs evaluation on all evaluation languaes at once
        inference_generation.main(**kwargs)

        logger.info(f"finished inference...")