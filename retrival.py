import os
import csv
import pickle
import time
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
import argparse

def parse_date(date_str):
    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %Y"
    ]
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt).date()
            return date_obj
        except ValueError:
            pass
    return None

def extract_dates(text):
    doc = nlp(text)
    dates = ""
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates += ent.text + " "
    dates = dates.strip()
    processed_dates=parse_date(dates)
    return processed_dates # 移除末尾的空格




class Retrieval:
    def __init__(self,d, model_name, question_list, triple_list,  embedding_size=768):
        self.model = SentenceTransformer(model_name)
        self.embedding_size = embedding_size
        self.question_list = question_list
        if d=='time':
            self.fact_list = [f'{f[0]} {f[1]} {f[2]} from {f[3]} to {f[4]}.' for f in triple_list]
        else:
            self.fact_list = [f'{f[0]} {f[1]} {f[2]} in {f[3]}.' for f in triple_list]
        self.full_time = [triple[3] for triple in triple_list]
        self.triplet_embeddings = None
        self.questions = []
        self.question_embedding = None
        self.index = None

    def build_faiss_index(self, n_clusters=1024, nprobe=128):
        quantizer = faiss.IndexFlatIP(self.embedding_size)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = nprobe
        return self.index

    def get_embedding(self,corpus_list):
        corpus_embeddings = self.model.encode(corpus_list, show_progress_bar=True, convert_to_numpy=True)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
        return corpus_embeddings

    def compute_similarity(self, n):
        self.question_embedding = self.get_embedding(self.question_list)
        self.triplet_embeddings = self.get_embedding(self.fact_list)
        index = self.build_faiss_index()
        index.train(self.triplet_embeddings)
        index.add(self.triplet_embeddings)
        distances, corpus_ids = self.index.search(self.question_embedding, n)
        return distances, corpus_ids


    def get_result(self, distances, corpus_ids, question_list, re_rank=False):
        result_list = []
        for i in range(len(corpus_ids)):
            if re_rank:
                result = self.re_rank_single_result(i, distances[i], corpus_ids[i], question_list)
            else:
                result = self.basic_result(i, distances[i], corpus_ids[i],question_list)
            # print(result)
            result_list.append(result)
        return result_list

    def re_rank_single_result(self, i, distances, corpus_ids, question_list):
        q = question_list[i]
        target_time = extract_dates(q)
        time_list = [10 for _ in range(len(self.full_time))]
        if target_time and target_time != "None":
            target_time = datetime.strptime(target_time, "%Y-%m-%d")
            self.adjust_time_scores(q, target_time, time_list)
        result = {'question': self.question_list[i]}
        hits = [{'corpus_id': id, 'score': score, 'final_score': score * 0.4 - time_list[id] * 0.6}
                for id, score in zip(corpus_ids, distances)]
        hits = sorted(hits, key=lambda x: x['final_score'], reverse=True)
        result['scores'] = [str(hit['score']) for hit in hits][:15]
        result['final_score'] = [str(hit['final_score']) for hit in hits][:15]
        result['triple'] = [self.triplet_list[hit['corpus_id']] for hit in hits]
        result['fact'] = [self.fact_list[hit['corpus_id']] for hit in hits]
        return result

    def adjust_time_scores(self, q, target_time, time_list):
        for idx, t in enumerate(self.full_time):
            time_difference = target_time - t
            days_difference = time_difference.days
            if 'before' in q:
                if 0 < days_difference < 16:
                    time_list[idx] = days_difference / 15
            elif 'after' in q:
                if -16 < days_difference < 0:
                    time_list[idx] = -days_difference / 15
            elif 'in' in q and days_difference == 0:
                time_list[idx] = 0

    def basic_result(self, i, distances, corpus_ids,question_list):
        result = {'question': question_list[i]}
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids, distances)]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        # result['fact'] = [self.corpus_sentences[hit['corpus_id']] for hit in hits]
        result['scores'] = [str(hit['score']) for hit in hits]
        result['triple'] = [self.triplet_list[hit['corpus_id']] for hit in hits]
        result['fact'] =  [self.fact_list[hit['corpus_id']] for hit in hits]
        return result

    def save_results(self, result_list, output_path):
        with open(output_path, "w", encoding='utf-8') as json_file:
            json.dump(result_list, json_file, indent=4)
