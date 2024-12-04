import json
from datetime import datetime,timedelta
import spacy
import en_core_web_sm
from retrival import Retrieval
nlp = en_core_web_sm.load()
import random
from tqdm import tqdm
import argparse

def parse_date(date_str):
    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %Y",
        "%Y"
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
    return processed_dates

def random_choice_except(target_list,nums,entity_list):
    options = [item for item in entity_list if item not in target_list]
    if options:
        return random.sample(options, nums)
    else:
        return None 

def generate_negative(postive,time, time_list, questions,entity_list,relation_list):
    question = questions['question']
    answers = questions['answers']
    random_number = random.randint(0, 10)
    negative = []
    time = postive[3]
    if question.__contains__("before") or question.__contains__("first"):
        new_time = time + timedelta(days=random_number)
        if new_time not in time_list:
            new_time = random_choice_except(time,1, time_list)
    elif question.__contains__("after") or question.__contains__("last"):
        new_time = time - timedelta(days=random_number)
        if new_time not in time_list:
            new_time = random_choice_except(time,1, time_list)
    else:
        raise ValueError()
    #Construct negative with incorrect time
    neg = " ".join(postive[:3]).replace('_', ' ')  + ' ' + str(new_time)
    negative.append(neg)
    #Construct negative with incorrect content
    new_relation = random_choice_except(postive[1], 1, relation_list)
    new_entity = random_choice_except(answers, 1, entity_list)
    neg = postive[0]  + ' ' + new_relation +' ' + new_entity + ' ' + str(time)
    negative.extend(neg)
    neg = postive[0] + ' ' + new_relation +' ' + new_entity + ' ' + str(new_time)
    negative.extend(neg)    
    return negative

def sort_by_date(item):
    return datetime.strptime(item[3], "%Y-%m-%d")

def calculate_time_difference(time_str1, time_str2):
    def to_datetime(time_str):
        if isinstance(time_str, str):
            return datetime.strptime(time_str, "%Y-%m-%d").date()
        return time_str

    time1 = to_datetime(time_str1)
    time2 = to_datetime(time_str2)

    return (time1 - time2).days


def main(args):

    with open(args.triplet_path, 'r') as file:
        triple_list = json.load(file)
    triple_list = [[item.replace('_', ' ') for item in sublist] for sublist in triple_list]

    with open(args.question_path, 'r') as file:
        question_list = json.load(file)
    print(len(question_list))

    #Retrieve facts based on posterior information
    corpu_list = [f"{item['question']} {str(item['answers'])}" for item in question_list]
    model_name = 'all-mpnet-base-v2'
    retriever = Retrieval(model_name, corpu_list, triple_list)
    distances, corpus_ids = retriever.compute_similarity(n=15)
    fact_list = retriever.get_result(distances, corpus_ids, question_list)

    with open(args.entity2id_path, 'r') as file:
        entity_dict = json.load(file)
    entity_dict = {key.replace('_', ' '): value for key, value in entity_dict.items()}

    with open(args.relation2id_path, 'r') as file:
        relation_dict = json.load(file)
    relation_dict = {key.replace('_', ' '): value for key, value in relation_dict.items()}

    time2id = {}
    with open(args.time2id_path, 'r') as file:
        time2id = json.load(file)

    result_list  = []
    for j in tqdm(range(len(question_list)), desc="Retrieving facts", unit="j"):
        question = question_list[j]['question']
        answers = question_list[j]['answers']
        triplets = fact_list[j]['triple']
        fact = fact_list[j]['fact']
        time = extract_dates(question)
        positive = ""
        positive_triplet = []
        negative = []
        if time:
            time= datetime.strptime(time, "%Y-%m-%d").date()
            for a in answers:
                if str(fact[i]).__contains__(a):
                    positive_triplet = triplets[i]
                    positive = fact[i]
                    break
            negative = generate_negative(positive,time, time2id.keys(),question_list[j],entity_dict.keys(),relation_dict.keys())
        elif str(question).__contains.__("last"):
            answer_list = []
            for i in range(len(triplets)):
                for a in answers:
                    if str(fact[i]).__contains__(a):
                        answer_list.append(triplets[i])
            sorted_data = sorted(answer_list, key=sort_by_date, reverse=True)
            if len(sorted_data)==0:
                j = j+1
                continue
            positive_triplet = sorted_data[-1]        
            positive = ' '.join(positive_triplet)
            negative = generate_negative(positive, time, time2id.keys(),question_list[j],entity_dict.keys(),relation_dict.keys())

        elif str(question).__contains__("first"):
            answer_list = []
            for i in range(len(triplets)):
                triplets[i][3] = triplets[i][3].replace('\n', '')
                for a in answers:
                    if str(fact[i]).__contains__(a):
                        answer_list.append(triplets[i])
            sorted_data = sorted(answer_list, key=sort_by_date, reverse=True)
            if len(sorted_data)==0:
                j = j+1
                continue
            positive_triplet = sorted_data[0]        
            positive = ' '.join(positive_triplet)
            negative = generate_negative(positive,time,time2id.keys(),question_list[j],entity_dict.keys(),relation_dict.keys())
        else:
            j = j+1
            continue

        if  len(negative)<3 or positive == "":
            j = j+1
            continue

        result = {
            'question': question,
            'answers': answers,
            'positive': positive,
            'negative': negative
        }
        
        result_list.append(result)

    print(f"Total results: {len(result_list)}")

    with open(args.output_path, "w", encoding='utf-8') as json_file:
        json.dump(result_list, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--triplet_path", type=str)
    argparser.add_argument("--output_path", type=str)
    argparser.add_argument("--question_path", type=str)
    argparser.add_argument("--datasets", type=str)
    argparser.add_argument("--entity2id_path", type=str)
    argparser.add_argument("--time2id_path", type=str)

    args = argparser.parse_args()
    main(args)