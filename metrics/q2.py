import argparse
import torch
import os
import numpy as np
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import pipeline


DEVICE = 'cuda' \
    if torch.cuda.is_available() else 'cpu'

"""
Re-implementation of Q^2 from: https://github.com/orhonovich/q-squared
"""

class QuestionGeneration():

    def __init__(self, **kwargs):
        self.tokenizer  = AutoTokenizer.from_pretrained(kwargs["qg_dir"])
        self.model      = AutoModelForSeq2SeqLM.from_pretrained(kwargs["qg_dir"]).to(DEVICE)


    def get_answer_candidates(self, text):
        text = str(text)
        model_dir = "davlan/xlm-roberta-base-wikiann-ner"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=True)
        model = AutoModelForTokenClassification.from_pretrained(model_dir, use_auth_token=True)
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        
        return [re.sub(r'^▁*', "", ent["word"]) for ent in nlp(text) if re.sub(r'^▁*', "", ent["word"]) != ""]


class QuestionAnswering():

    def __init__(self, **kwargs):
        self.tokenizer  = AutoTokenizer.from_pretrained(kwargs["qa_dir"])
        self.model      = AutoModelForSeq2SeqLM.from_pretrained(kwargs["qa_dir"]).to(DEVICE)
    
    def get_response(self, question, context, max_length=32):
        input_text = f"question: {question} context: {context}"
        features = self.tokenizer([input_text], return_tensors="pt")
        output = self.model.generate(input_ids=features["input_ids"].to(DEVICE),
                                     attention_mask=features["attention_mask"].to(DEVICE),
                                     max_new_tokens=max_length
                                     )
        return re.sub(r"<pad>\s*|s*</s>", "", self.tokenizer.decode(output[0]))
    

    def get_span(self, return_dict:bool=False, **input_ids):

        start_scores, end_scores = self.model(**input_ids, return_dict=return_dict)
        start   = torch.argmax(start_scores, dim=-1)
        end     = torch.argmax(end_scores, dim=-1)

        return start, end


def main(**kwargs):
    os.environ['TRANSFORMERS_CACHE'] = kwargs["dataset_dir"]
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"
    qg = QuestionGeneration(**kwargs)
    qa = QuestionAnswering(**kwargs)

    qa.model.eval()
    qg.model.eval()
    with torch.no_grad():
        df = pd.read_json(os.path.join(kwargs["dataset_dir"], "outputStandardFileBaseline_fr_vi.json"), lines=True) #.sample(200, random_state=42)
        df = df[df['lang'].isin(kwargs["eval_lang"])]

        # def escape_pipe_in_list(lst):
        #     return [item.replace('|', '-') for item in lst]

    # Apply the function to the DataFrame column
        
        if kwargs["sample_size"] is not None:
            df = df.sample(70, random_state=42)
        
        # md_table = df.to_markdown(index=False)
        # with open(os.path.join(kwargs['dataset_dir'],f'{"_".join(kwargs["eval_lang"])}_samplesize{kwargs["sample_size"]}.md'), 'w') as f:
        #     f.write(md_table)


        valid_questions = []
        valid_cands = []
        knowledge_answers = []
        
        for row in tqdm(df.itertuples(), total=len(df)):
            orig_answer      = row.response
            orig_answer      = re.sub(r"^<response>\s*", "", orig_answer)
            knowledge        = row.positive
            # language    = row.lang

            candidates = qg.get_answer_candidates(text=orig_answer)
            print(f"{candidates=}")
            if len(candidates) == 0: # no (grouped) entities found
                continue

            input_texts = []
            for cand in candidates:
                input_texts.append(f"answer: {orig_answer} context: {cand}")
            
            features = qg.tokenizer(input_texts, return_tensors='pt', padding=True).to(DEVICE)
            beam_outputs = qg.model.generate(**features, num_beams=2, num_return_sequences=1, 
                                            temperature=0.8, 
                                             max_new_tokens=512)

            questions = list(set([output.replace("<pad>", "").strip() for output in qg.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)]))
                
            knowledge    = knowledge.split("//", 1)[0].strip() # exclude document structure, e.g., title
            for question in questions:
                valid_cands_tmp  = qa.get_response(question=question, context=orig_answer)
                if valid_cands_tmp not in candidates: # response r does not match answer span a^r_i, used for question generation
                    valid_questions     += [""]
                    knowledge_answers   += [knowledge]
                    valid_cands         += [orig_answer]
                    # continue
                else:
                    valid_questions     += [question]       # * len(knowledge)
                    valid_cands         += [valid_cands_tmp]    # * len(knowledge)     
                    knowledge_answers   += [qa.get_response(question=question, context=knowledge)]
        
                print(f"{valid_questions[-1:]=}; {knowledge_answers[-1:]=}; {valid_cands[-1:]=}")

        del qa.model
        del qg.model

    # NLI step
    model = AutoModelForSequenceClassification.from_pretrained(kwargs["nli_dir"]).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(kwargs["nli_dir"])

    model.eval()
    with torch.no_grad():
        scores = []
        premises = []
        for premise_tmp, hypothesis, val_question in tqdm(zip(valid_cands, knowledge_answers, valid_questions)):
            premise     = f"{val_question} {premise_tmp}"
            hypothesis  = f"{val_question} {hypothesis}"

            input   = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
            outputs = model(input["input_ids"].to(DEVICE))
            
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

            premises += [premise_tmp]

            if predicted_label == 0:    # entailment
                scores.append(1.0)
            elif predicted_label == 1:  # neutral
                scores.append(0.5)
            else:                       # contradiction
                scores.append(0.0)

        q2_df = pd.DataFrame({"premise": premises, "q2":scores})
        q2_df = q2_df.groupby('premise')['q2'].median().reset_index()
        print(q2_df)
        
        df["response"] = df["response"].str.replace(r"^<response>\s*", "", regex=True)
        print(df)
        df = df.merge(q2_df, left_on='response', right_on='premise', how='inner')

        print(df)
        df = df.sample(kwargs["sample_size"], random_state=42)

        print(df)
        df['positive'] = df['positive'].str.replace('|', '\|')
        md_table = df[["query", "positive", "response", "q2"]].to_markdown(index=False)
        with open(os.path.join(kwargs['dataset_dir'],f'{"_".join(kwargs["eval_lang"])}_samplesize{kwargs["sample_size"]}.md'), 'w') as f:
            f.write(md_table)

        print(f"{scores=}")
        print(f"{np.mean(scores)=}")

        del model
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--dataset-dir", help= "Specifiy dataset dir to save model to", type= str, default= "./0_baseline")
    parser.add_argument("---qg-dir", help= "Specifiy question generation model dir", type= str, default= "Narrativa/mT5-base-finetuned-tydiQA-question-generation")
    parser.add_argument("---qa-dir", help= "Specifiy question answer model dir", type= str, default= "Narrativa/mT5-base-finetuned-tydiQA-xqa")
    parser.add_argument("--nli-dir", help= "Specify NLI model dir", type=str, default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--eval-lang", help= "Specify languages to evaluate Q^2 metric", type=eval, default="['fr', 'vi']")
    parser.add_argument("--sample-size", help= "Calculate q^2 on specified number of samples", type= int, default=None)
    args, _ = parser.parse_known_args()

    main(**vars(args))