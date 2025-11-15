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
from peft import PeftModel
import en_core_web_sm
from tqdm import tqdm
import argparse

# Load the spaCy English model for NLP tasks like named entity recognition.
nlp = en_core_web_sm.load()

def parse_date(date_str):
    """
    Parses a date string into a datetime.date object.
    It tries multiple common date formats.

    Args:
        date_str (str): The string representation of the date.

    Returns:
        datetime.date or None: The parsed date object, or None if parsing fails for all formats.
    """
    formats = [
        "%Y-%m-%d",  # e.g., "2023-12-25"
        "%d %B %Y",  # e.g., "25 December 2023"
        "%B %Y",     # e.g., "December 2023"
        "%Y"         # e.g., "2023"
    ]
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt).date()
            return date_obj
        except ValueError:
            pass
    return None

def extract_dates(text):
    """
    Extracts date entities from a given text using spaCy.

    Args:
        text (str): The input text to process.

    Returns:
        datetime.date or None: The first successfully parsed date object found in the text.
    """
    doc = nlp(text)
    dates = ""
    # Iterate through entities recognized by spaCy.
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates += ent.text + " "
    dates = dates.strip()
    processed_dates = parse_date(dates)
    return processed_dates

def date_to_id(date_obj):
    """
    Converts a date object to a unique integer ID.
    The ID is the number of days since a fixed base date (1005-01-01).

    Args:
        date_obj (datetime.date): The date to convert.

    Returns:
        int: The integer ID representing the date.
    """
    base_date = datetime(1005, 1, 1).date()
    return (date_obj - base_date).days


class Retrieval:
    """
    A class for retrieving relevant facts (triplets) from a knowledge base
    for a given list of questions using semantic similarity and optional re-ranking.
    """
    def __init__(self, model_name, question_list, triple_list, id_list, time_list, embedding_size=768):
        """
        Initializes the Retrieval class.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
            question_list (list): A list of question strings.
            triple_list (list): A list of fact triplets, where each triplet is a list [head, relation, tail, time].
            id_list (list): A list of triplet IDs corresponding to triple_list.
            time_list (list): A list of target times extracted from questions.
            embedding_size (int): The dimension of the sentence embeddings.
        """
        self.model = SentenceTransformer(model_name, device="cuda")
        self.embedding_size = embedding_size
        self.time_list = time_list
        self.question_list = question_list
        self.triple_list = triple_list
        self.triplet_id_list = [[triple[0], triple[1], triple[2], triple[3]] for triple in id_list]
        self.full_time = [datetime.strptime(triple[3], "%Y-%m-%d").date() for triple in triple_list]
        
        # Convert triplets into natural language sentences for embedding.
        self.fact_list = [f'{f[0]} {f[1]} {f[2]} in {f[3]}.' for f in triple_list]

        self.triplet_embeddings = None
        self.question_embedding = None
        self.index = None

    def build_faiss_index(self, n_clusters=2, nprobe=128):
        """
        Builds a FAISS index for efficient similarity search.
        Uses IndexIVFFlat for faster search on large datasets.

        Args:
            n_clusters (int): The number of Voronoi cells for the IVF index.
            nprobe (int): The number of nearby cells to search during lookup.
        
        Returns:
            faiss.Index: The configured FAISS index.
        """
        quantizer = faiss.IndexFlatIP(self.embedding_size)  # Base index using Inner Product
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = nprobe
        return self.index

    def get_embedding(self, corpus_list):
        """
        Encodes a list of texts into normalized embeddings.

        Args:
            corpus_list (list): A list of strings to encode.

        Returns:
            np.ndarray: A 2D numpy array of normalized embeddings.
        """
        corpus_embeddings = self.model.encode(corpus_list, show_progress_bar=True, convert_to_numpy=True, batch_size=512)
        # Normalize embeddings to unit length for inner product similarity.
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
        return corpus_embeddings

    def compute_similarity(self, n):
        """
        Computes similarity between questions and facts, and retrieves the top n facts for each question.

        Args:
            n (int): The number of top similar facts to retrieve for each question.

        Returns:
            tuple: A tuple containing (distances, corpus_ids) for the search results.
        """
        print("Generating question embeddings...")
        self.question_embedding = self.get_embedding(self.question_list)
        
        print("Generating fact embeddings...")
        self.triplet_embeddings = self.get_embedding(self.fact_list)

        print("Building and training FAISS index...")
        index = self.build_faiss_index()
        index.train(self.triplet_embeddings)
        index.add(self.triplet_embeddings)
        
        print(f"Searching for top {n} facts...")
        distances, corpus_ids = self.index.search(self.question_embedding, n)
        return distances, corpus_ids

    def get_result(self, distances, corpus_ids, question_list, re_rank=False):
        """
        Formats the retrieval results. Can perform re-ranking if specified.

        Args:
            distances (np.ndarray): Similarity scores from FAISS.
            corpus_ids (np.ndarray): Indices of retrieved facts from FAISS.
            question_list (list): The original list of question dicts.
            re_rank (bool): If True, apply the re-ranking logic.

        Returns:
            list: A list of dictionaries, each containing results for one question.
        """
        if re_rank:
            # Apply advanced re-ranking that considers time matching.
            return [
                self.re_rank_single_result(i, distances[i], corpus_ids[i], question_list[i]['question'])
                for i in tqdm(range(len(corpus_ids)), desc="Reranking")
            ]
        else:
            # Use basic ranking based only on semantic similarity.
            return [
                self.basic_result(i, distances[i], corpus_ids[i])
                for i in tqdm(range(len(corpus_ids)), desc="Basic Ranking")
            ]

    def re_rank_single_result(self, i, distances, corpus_ids, q, alpha=0.8):
        """
        Re-ranks a single question's results by fusing semantic similarity with a time-matching score.

        Args:
            i (int): The index of the current question.
            distances (list): List of similarity scores.
            corpus_ids (list): List of retrieved fact IDs.
            q (str): The question text.
            alpha (float): The weight for the semantic similarity score. (1-alpha) is the weight for the time score.

        Returns:
            dict: A dictionary containing the re-ranked results for the question.
        """
        target_time = self.time_list[i]
        question = self.question_list[i]
        time_scores = {}

        # ======= 1. Extract temporal keywords from the question ========
        time_keywords = []
        if q:
            q_lower = q.lower()
            for kw in ['before', 'after', 'in', 'on', 'first', 'last']:
                if kw in q_lower:
                    time_keywords.append(kw)
            if 'on' in time_keywords and 'in' not in time_keywords:
                time_keywords.remove('on')
                time_keywords.append('in') # Normalize 'on' to 'in'

        # ======= 2. Parse the target time from question metadata ========
        parsed_target = None
        if target_time and isinstance(target_time, str) and target_time != "None":
            # Handle partial dates
            if len(target_time) == 4: target_time += '-01-01' # YYYY -> YYYY-01-01
            elif len(target_time) == 7: target_time += '-01'   # YYYY-MM -> YYYY-MM-01
            parsed_target = datetime.strptime(target_time, "%Y-%m-%d").date()
        elif isinstance(target_time, datetime):
            parsed_target = target_time.date()

        # ======= 3. Calculate time-matching scores (higher is better) ========
        if parsed_target:
            for cid in corpus_ids:
                corpus_date = self.full_time[cid]
                days_diff = (parsed_target - corpus_date).days
                
                if 'before' in time_keywords:
                    if 0 < days_diff < 30: # Fact date is shortly before target date
                        time_scores[cid] = 1.0 - (days_diff / 30) # Closer is better
                    elif days_diff < 0:
                        time_scores[cid] = -100  # Heavy penalty for wrong direction
                    else:
                        time_scores[cid] = -1 # Penalty
                elif 'after' in time_keywords:
                    if -30 < days_diff < 0: # Fact date is shortly after target date
                        time_scores[cid] = 1.0 - (-days_diff / 30) # Closer is better
                    elif days_diff > 0:
                        time_scores[cid] = -100  # Heavy penalty for wrong direction
                    else:
                        time_scores[cid] = -1 # Penalty
                elif 'in' in time_keywords:
                    time_scores[cid] = 1.0 if days_diff == 0 else -100 # Exact match or heavy penalty
        else:
            # Default neutral score if no time is found.
            for cid in corpus_ids:
                time_scores[cid] = 0

        # ======= 4. Calculate the fused score ========
        hits = []
        for cid, sim in zip(corpus_ids, distances):
            time_match = time_scores.get(cid, 0.5) # Default score if not calculated
            final_score = alpha * sim + (1 - alpha) * time_match
            hits.append({
                'corpus_id': cid,
                'score': sim,
                'final_score': final_score,
                'time': self.full_time[cid]
            })

        # ======= 5. Sort results based on the question type ========
        if 'first' in time_keywords or 'last' in time_keywords:
            # For 'first'/'last', sort chronologically.
            reverse = 'last' in time_keywords
            hits = sorted(hits, key=lambda x: x['final_score'], reverse=True)[:20] # Pre-filter by score
            
            with_time = [h for h in hits if h.get('time') is not None]
            without_time = [h for h in hits if h.get('time') is None]
            
            with_time.sort(key=lambda x: x['time'], reverse=reverse)
            hits = with_time + without_time
        else:
            # For other questions, sort by the combined final score.
            hits.sort(key=lambda x: x['final_score'], reverse=True)

        top_hits = hits[:20]

        # ======= 6. Assemble the final result dictionary ========
        result = {
            'question': question,
            'fact': [self.fact_list[hit['corpus_id']] for hit in top_hits],
            'scores': [str(hit['score']) for hit in top_hits],
            'final_score': [str(hit['final_score']) for hit in top_hits],
            'triple': [self.triple_list[hit['corpus_id']] for hit in top_hits]
        }
        return result

    def basic_result(self, i, distances, corpus_ids):
        """
        Formats the retrieval results based only on the initial semantic similarity score.

        Args:
            i (int): The index of the current question.
            distances (list): List of similarity scores.
            corpus_ids (list): List of retrieved fact IDs.

        Returns:
            dict: A dictionary containing the top results for the question.
        """
        result = {'question': self.question_list[i]}
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids, distances)]
        
        # Sort by the initial similarity score in descending order.
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        top_hits = hits[:20] # Take the top 20 hits

        result['fact'] = [self.fact_list[hit['corpus_id']] for hit in top_hits]
        result['scores'] = [str(hit['score']) for hit in top_hits]
        result['triple'] = [self.triple_list[hit['corpus_id']] for hit in top_hits]
        return result

    def save_results(self, result_list, output_path):
        """
        Saves the list of results to a JSON file.

        Args:
            result_list (list): The list of result dictionaries to save.
            output_path (str): The path to the output JSON file.
        """
        print(f"Saving results to {output_path}...")
        with open(output_path, "w", encoding='utf-8') as json_file:
            json.dump(result_list, json_file, indent=4, ensure_ascii=False)
        print("Done.")