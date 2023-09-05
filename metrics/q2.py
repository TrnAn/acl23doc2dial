import argparse
import torch
import os
from operator import itemgetter
import numpy as np
import spacy
import re
import torch
import string
import pandas as pd
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline
# from nltk.tokenize import RegexpTokenizer
from collections import defaultdict


nlp_en = spacy.load("en_core_web_sm")
nlp_vi = spacy.load("xx_ent_wiki_sm")
nlp_fr = spacy.load("fr_core_news_sm")

NLP = {"fr": nlp_fr,
        "vi": nlp_vi,
        "en": nlp_en}

NLP_DD = defaultdict(lambda: nlp_en, NLP)
ALL_STOPWORDS = [nlp_func.Defaults.stop_words for nlp_func in NLP.values()]

DEVICE = 'cuda' \
    if torch.cuda.is_available() else 'cpu'

"""
Re-implementation of Q^2 from: https://github.com/orhonovich/q-squared
"""
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stopword_pattern = r'\b(?:' + '|'.join(re.escape(stopword) for stopword in ALL_STOPWORDS) + r')\b'
    text = re.sub(stopword_pattern, ' ', text)
    return re.sub(' +', ' ', text).strip()

class QuestionGeneration():

    def __init__(self, **kwargs):
        self.tokenizer  = AutoTokenizer.from_pretrained(kwargs["qg_dir"])
        self.model      = AutoModelForSeq2SeqLM.from_pretrained(kwargs["qg_dir"]).to(DEVICE)


    def get_answer_candidates(self, text, lang):
        text = str(text)

        doc = NLP_DD[lang](text)
        cand_dict           = {ent.text: ent.text.lower() for ent in list(doc.ents)}  # unique response candidates
        candidates  = list(cand_dict.keys())

        if lang not in ["vi"]:
            noun_chunks_dict    = {chunks.text: chunks.text.lower() for chunks in doc.noun_chunks} # unique response candidates                
            new_cand_keys = set(noun_chunks_dict.keys()) - set(cand_dict.keys())

            candidates.append(itemgetter(*new_cand_keys)(noun_chunks_dict))
    
        # for chunk in noun_chunks:
        #     found = False
        #     for cand in candidates:
        #         if chunk.text.lower() == cand.lower():
        #             found = True
        #     if not found:
        #         candidates.append(chunk.text)
        # candidates = [cand for cand in candidates if cand.lower() != 'i']

        return candidates

class QuestionAnswering():

    def __init__(self, **kwargs):
        self.tokenizer  = AutoTokenizer.from_pretrained(kwargs["qa_dir"])
        self.model      = AutoModelForQuestionAnswering.from_pretrained(kwargs["qa_dir"]).to(DEVICE)
    

    def get_span(self, return_dict:bool=False, **input_ids):

        start_scores, end_scores = self.model(**input_ids, return_dict=return_dict)
        start   = torch.argmax(start_scores, dim=-1)
        end     = torch.argmax(end_scores, dim=-1)

        return start, end


def main(is_dstc:bool=True, **kwargs):
    os.environ['TRANSFORMERS_CACHE'] = kwargs["dataset_dir"]
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"
    qg = QuestionGeneration(**kwargs)
    qa = QuestionAnswering(**kwargs)

    qa.model.eval()
    qg.model.eval()
    with torch.no_grad():
        df = pd.read_json(os.path.join(kwargs["dataset_dir"], "outputStandardFileBaseline_fr_vi.json"), lines=True)
        # df["knowledge"] = df['passages'].agg(lambda x: ', '.join(x), axis=1)

        # qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        # qg_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap").to("cuda:0")
        # f1 = 0
        # num_questions = 0
        valid_questions = []
        valid_cands = []
        knowledge_answers = []
        scores = []
        
        for idx, row in tqdm(df.iterrows()):
            orig_answer      = row["response"]
            
            knowledge   = row["passages"]
            language    = row["lang"]

            valid_questions.append([])
            valid_cands.append([])
            knowledge_answers.append([])
            candidates = qg.get_answer_candidates(text=orig_answer, lang=language)
            # print(f"{candidates=}")
            if len(candidates) <= 0:
                print("skip")
                continue
            # print(f"{candidates=}")

            # if len(candidates) > 0:
            input_texts = []
            all_questions = []
            for cand in candidates:
                input_texts.append(f"answer: {orig_answer} context: {cand} </s>")
            
            features = qg.tokenizer(input_texts, return_tensors='pt', padding=True).to(DEVICE)
            beam_outputs = qg.model.generate(**features, num_beams=2, num_return_sequences=1)

            questions = list(set([output.replace("<pad>", "").strip() for output in qg.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)]))
            print(f"{questions=} -> {orig_answer=} === {candidates=}")
            qa_inputs = qa.tokenizer(questions, [orig_answer]*len(questions), add_special_tokens=True, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            for knowledge_sample in knowledge:
                # for knowledge_sample in knowledge_samples:
                # if is_dstc:
                #     knowledge_sample = knowledge_sample["text"].split("A: ")[-1]
                # else:
                    # knowledge_sample = knowledge_sample["text"]
                # knowledge_sample = knowledge_sample#[i]["text"]
                # print(f"{knowledge_sample=}")
                qa_inputs_knowledge = qa.tokenizer(questions, [knowledge_sample]*len(questions), add_special_tokens=True, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
                # answer_start_scores, answer_end_scores = qa_model(**qa_inputs, return_dict=False)
                # answer_start_scores_knowledge, answer_end_scores_knowledge = qa_model(**qa_inputs_knowledge, return_dict=False)
                answer_start, answer_end = qa.get_span(**qa_inputs, return_dict=False)
                # print(f"{answer_start=} {answer_end=}")
                answer_start_knowledge, answer_end_knowledge = qa.get_span(**qa_inputs_knowledge, return_dict=False)
                # print(f"{answer_start_knowledge=} {answer_end_knowledge=}")
                # answer_start = torch.argmax(answer_start_scores, dim=-1)
                # answer_end = torch.argmax(answer_end_scores, dim=-1) + 1

                # answer_start_knowledge = torch.argmax(answer_start_scores, dim=-1) # answer_start_knowledge = torch.argmax(answer_start_scores_knowledge, dim=-1)
                # answer_end_knowledge = torch.argmax(answer_end_scores, dim=-1) + 1 # answer_end_knowledge = torch.argmax(answer_end_scores_knowledge, dim=-1) + 1
                # answer_start_knowledge = torch.argmax(answer_start_scores_knowledge, dim=-1)
                # answer_end_knowledge = torch.argmax(answer_end_scores_knowledge, dim=-1) + 1

                input_ids = qa_inputs["input_ids"].cpu()
                input_ids_knowledge = qa_inputs_knowledge["input_ids"].cpu()


                for i, (start, end) in enumerate(zip(answer_start.cpu().numpy(), answer_end.cpu().numpy())):
                    answer = qa.tokenizer.convert_tokens_to_string(qa.tokenizer.convert_ids_to_tokens(input_ids[i][start:end]))
                    # print(f"{answer=}")
                    # if clean_text(answer) == clean_text(candidates[i]):
                    valid_questions[idx].append(questions[i])
                    # valid_cands.append(candidates[i])
                    # valid_cands[idx].append(candidates[i])
                    valid_cands[idx].append(answer)
                    
                for i, (start, end) in enumerate(zip(answer_start_knowledge.cpu().numpy(), answer_end_knowledge.cpu().numpy())):
                    answer_knowledge = qa.tokenizer.convert_tokens_to_string(qa.tokenizer.convert_ids_to_tokens(input_ids_knowledge[i][start:end]))
                    knowledge_answers[idx].append(answer_knowledge)

        del qa.model
        del qg.model
        # classifier = pipeline('zero-shot-classification', model='microsoft/deberta-v2-large-mnli')
        model = AutoModelForSequenceClassification.from_pretrained(kwargs["nli_dir"]).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(kwargs["nli_dir"])

        for answers_knowledge, cands, questions in tqdm(zip(
            knowledge_answers, valid_cands, valid_questions
        )):
            for answer_knowledge, cand, question in zip(
                answers_knowledge, cands, questions
            ):
            # premise = question + ' ' + answer_knowledge + '.'
            # hypothesis = question + ' ' + cand + '.'
                local_scores = []
                if cand == "": # qa generated an empty answer
                    continue
                input_ = f"[CLS] {question} {answer_knowledge} [SEP] {question} {cand} [SEP]"
      
                
                inputs = tokenizer(input_, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
                outputs = model(**inputs)
                best = torch.argmax(outputs.logits)
                if best == 2:
                    local_scores.append(1.0)
                elif best == 1:
                    local_scores.append(0.5)
                else:
                    local_scores.append(0.0)

            if len(local_scores) > 0: # applies if answer is no empty
                scores.append(float(np.mean(local_scores)))

        print(f"{np.mean(scores)=}; {scores=}")

        del model
        # return np.mean(scores), scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    # parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    parser.add_argument("--dataset-dir", help= "Specifiy dataset dir to save model to", type= str, default= "./0_baseline")
    parser.add_argument("---qg-dir", help= "Specifiy question generation model dir", type= str, default= "Narrativa/mT5-base-finetuned-tydiQA-question-generation")
    parser.add_argument("---qa-dir", help= "Specifiy question answer model dir", type= str, default= "deepset/xlm-roberta-base-squad2")
    parser.add_argument("--nli-dir", help= "Specify NLI model dir", type=str, default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    args, _ = parser.parse_known_args()

    main(**vars(args))