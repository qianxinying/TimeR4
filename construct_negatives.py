import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import argparse
import spacy
from retrival import Retrieval  # Assuming the retrival module is correctly implemented

class NegativeSampler:
    """
    A class to generate negative samples for temporal question answering.
    """

    def __init__(self, model_name='en_core_web_sm'):
        """
        Initializes the NegativeSampler.

        Args:
            model_name (str): The name of the spaCy model to load.
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"spaCy model '{model_name}' not found.")
            print("Please run 'python -m spacy download en_core_web_sm' to download it.")
            exit()

    @staticmethod
    def parse_date(date_str):
        """
        Parses a string into a date object, supporting multiple formats.

        Args:
            date_str (str): The date string.

        Returns:
            datetime.date or None: The parsed date object if successful, otherwise None.
        """
        if not date_str:
            return None
        formats = ["%Y-%m-%d", "%d %B %Y", "%B %Y", "%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None

    def extract_dates(self, text):
        """
        Extracts dates from a given text.

        Args:
            text (str): The input text.

        Returns:
            datetime.date or None: The extracted and parsed date object if successful, otherwise None.
        """
        doc = self.nlp(text)
        dates_str = " ".join([ent.text for ent in doc.ents if ent.label_ == "DATE"])
        return self.parse_date(dates_str.strip())

    @staticmethod
    def _random_choice_except(target_list, nums, entity_list):
        """
        Randomly selects a specified number of items from a list, excluding items from a target list.
        
        Args:
            target_list (list): A list of items to exclude.
            nums (int): The number of items to select.
            entity_list (list): The list to select items from.
        
        Returns:
            list or None: A list of randomly selected items, or None if no options are available.
        """
        options = [item for item in entity_list if item not in target_list]
        return random.sample(options, nums) if options else None

    def _generate_negatives(self, positive_triplet, question_info, time_list, entity_list, relation_list):
        """
        Generates negative samples for a given positive triplet.

        Args:
            positive_triplet (list): The correct fact triplet [head, relation, tail, time].
            question_info (dict): A dictionary containing the question and answers.
            time_list (list): A list of all possible timestamps.
            entity_list (list): A list of all possible entities.
            relation_list (list): A list of all possible relations.

        Returns:
            list: A list of generated negative triplets.
        """
        question = question_info['question'].lower()
        answers = question_info['answers']
        negatives = []

        try:
            time = datetime.strptime(positive_triplet[3], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            # Cannot generate time-based negatives if the positive triplet's time is invalid.
            return [] 

        # 1. Generate negative sample by altering the time.
        random_day_delta = timedelta(days=random.randint(1, 10))
        new_time_str = None

        if "before" in question or "first" in question:
            new_time = time + random_day_delta
        elif "after" in question or "last" in question:
            new_time = time - random_day_delta
        else:
            # For questions without clear temporal direction, pick a random time.
            new_time = None
        
        # If a new time was generated, check if it's a valid one.
        if new_time and new_time.strftime("%Y-%m-%d") not in time_list:
             # If the randomly generated time is not in the list, pick one from the list.
            new_time_str_list = self._random_choice_except([time.strftime("%Y-%m-%d")], 1, time_list)
            if new_time_str_list:
                new_time_str = new_time_str_list[0]
                negatives.append([positive_triplet[0], positive_triplet[1], positive_triplet[2], new_time_str])
        
        # 2. Generate negative samples by altering the relation and entity.
        new_relation = self._random_choice_except([positive_triplet[1]], 1, relation_list)
        # Choose a new entity that is not in the answer list.
        new_entity = self._random_choice_except(answers, 1, entity_list)

        if new_relation and new_entity:
            # Replace relation and entity, keep original time.
            negatives.append([positive_triplet[0], new_relation[0], new_entity[0], positive_triplet[3]])
            # Replace relation, entity, and time.
            if new_time_str:
                negatives.append([positive_triplet[0], new_relation[0], new_entity[0], new_time_str])

        return negatives
        
    @staticmethod
    def _sort_by_date(item):
        """Helper function to sort triplets by date."""
        try:
            return datetime.strptime(item[3], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            # For invalid dates, treat them as the earliest possible time for stable sorting.
            return datetime.min.date() 

def load_data(file_path, is_json=False):
    """
    Loads data from a file.

    Args:
        file_path (Path): The path to the file.
        is_json (bool): True if the file is in JSON format.

    Returns:
        list or dict: The loaded data.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    with file_path.open('r', encoding='utf-8') as f:
        if is_json:
            return json.load(f)
        else:
            return [line.strip().split('\t') for line in f]

def main(args):
    """
    Main execution function.
    """
    # --- 1. Load Data ---
    try:
        fact_list = load_data(Path(args.fact_path))
        # Replace underscores with spaces for better matching
        fact_list = [[item.replace('_', ' ') for item in sublist] for sublist in fact_list]
        fact_list = fact_list[:100]
        id_list = load_data(Path(args.triplet_id_path))
        question_list = load_data(Path(args.question_path), is_json=True)
        question_list = question_list[:10]
        entity_dict = load_data(Path(args.entity2id_path), is_json=True)
        entity_dict = {key.replace('_', ' '): value for key, value in entity_dict.items()}
        relation_dict = load_data(Path(args.relation2id_path), is_json=True)
        relation_dict = {key.replace('_', ' '): value for key, value in relation_dict.items()}
        time2id = load_data(Path(args.time2id_path), is_json=True)
    except FileNotFoundError as e:
        print(e)
        return

    # --- 2. Fact Retrieval ---
    print("Starting fact retrieval...")
    corpu_list = [f"{item['question']} {item['answers']}" for item in question_list]
    retriever = Retrieval(args.model_name, corpu_list, fact_list,id_list,None)
    distances, corpus_ids = retriever.compute_similarity(n=15)
    retrieved_facts = retriever.get_result(distances, corpus_ids, question_list)

    # --- 3. Generate Negative Samples ---
    sampler = NegativeSampler()
    result_list = []
    # Pre-list keys for faster random sampling
    entity_keys = list(entity_dict.keys())
    relation_keys = list(relation_dict.keys())
    time_keys = list(time2id.keys())

    print("Generating negative samples...")
    for i, question_info in enumerate(tqdm(question_list, desc="Processing questions")):
        question_text = question_info['question'].lower()
        answers = question_info['answers']
        triplets = retrieved_facts[i]['triple']
        
        positive_triplet = None
        
        # Determine the positive triplet based on the question type
        if "first" in question_text:
            sorted_triplets = sorted(triplets, key=sampler._sort_by_date)
            if sorted_triplets:
                positive_triplet = sorted_triplets[0]
        elif "last" in question_text:
            sorted_triplets = sorted(triplets, key=sampler._sort_by_date, reverse=True)
            if sorted_triplets:
                positive_triplet = sorted_triplets[0]
        else: # For other questions, default to the most relevant retrieved triplet
            if triplets:
                positive_triplet = triplets[0]

        if not positive_triplet:
            continue # Skip if no positive triplet can be identified

        negatives = sampler._generate_negatives(positive_triplet, question_info, time_keys, entity_keys, relation_keys)
        
        # Convert triplets to their corresponding IDs
        positive_id = [
            entity_dict.get(positive_triplet[0], -1),
            relation_dict.get(positive_triplet[1], -1),
            entity_dict.get(positive_triplet[2], -1),
            time2id.get(positive_triplet[3], -1)
        ]
        
        negative_ids = [
            [entity_dict.get(h, -1), relation_dict.get(r, -1), entity_dict.get(t, -1), time2id.get(d, -1)]
            for h, r, t, d in negatives
        ]

        result_list.append({
            'question': question_info['question'],
            'answers': answers,
            'positive': positive_triplet,
            'positive_id': positive_id,
            'negative': negatives,
            'negative_id': negative_ids
        })

    # --- 4. Save Results ---
    print(f"Total results generated: {len(result_list)}")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    with output_path.open("w", encoding='utf-8') as f:
        json.dump(result_list, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate negative samples for temporal question answering tasks.")
    parser.add_argument("--fact_path", type=str, default='timeR4/datasets/MultiTQ/kg/full.txt', help="Path to the knowledge base fact file.")
    parser.add_argument("--triplet_id_path", type=str, default='timeR4/datasets/MultiTQ/kg/full_id.txt', help="Path to the fact ID file.")
    parser.add_argument("--question_path", type=str, default='timeR4/datasets/MultiTQ/questions/train.json', help="Path to the question file.")
    parser.add_argument("--output_path", type=str, default='timeR4/datasets/MultiTQ/questions/negatives.json', help="Path to the output file.")
    parser.add_argument("--entity2id_path", type=str, default='timeR4/datasets/MultiTQ/kg/entity2id.json', help="Path to the entity-to-ID mapping file.")
    parser.add_argument("--relation2id_path", type=str, default='timeR4/datasets/MultiTQ/kg/relation2id.json', help="Path to the relation-to-ID mapping file.")
    parser.add_argument("--time2id_path", type=str, default='timeR4/datasets/MultiTQ/kg/ts2id.json', help="Path to the time-to-ID mapping file.")
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-mpnet-base-v2', help="Name or path of the sentence-transformer model for retrieval.")
    
    args = parser.parse_args()
    main(args)