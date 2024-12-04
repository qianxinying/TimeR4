import os
import csv
import pickle
import time
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
from retrival import Retrieval

os.environ['OPENAI_BASE_URL'] = 'YOUR_OPENAI_URL'
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_KEY'

#retrieval
def retrieve(d,model_name, question_list, triple_list):
    retriever = Retrieval(d,model_name, question_list, triple_list)
    distances, corpus_ids = retriever.compute_similarity(n=15)
    fact_list = retriever.get_result(distances, corpus_ids, question_list,re_rank=False)
    return fact_list

#rewrite
def gpt_chat_completion(**kwargs):
    backoff_time = 1
    while True:
        try:
            client = OpenAI()
            # client = Client()
            response = client.chat.completions.create(**kwargs)
            if type(response) == str:
                return response_parser(response)
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(e)
            # print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
            time.sleep(backoff_time)
            backoff_time *= 1.5

def response_parser(response):
    response = response.strip().split("data: ")[1:]
    result = ''
    for r in response:
        if r == '[DONE]':
            break
        delta = json.loads(r)['choices'][0]['delta']
        if 'content' in delta:
            result += delta['content']
    return result

def rewrite(fact_list,question_list):
    for i in tqdm(range(len(question_list)), desc="Processing data", unit="i"):
        question = question_list[i]['question']
        qlabel = question_list[i]['type']
        if "Implicit" in qlabel:
            fact = fact_list[i][0]
            prompt = f"Replace facts in questions with explicit information from provided facts without any explanation.If you are not sure about the answer, output the original question. For instance, from the fact \"Herman Van Rompuy replaced by Donald Tusk from 2009 to 2014.\" modify the question \"what was donald tusk's position before he was replaced by herman van rompuy?\" to \what was donald tusk's position before 2009?\" Here is your turn: Question: {question} Fact:{fact}."
            print(question)
            messages = [{"role": "user", "content": prompt},]
            response = gpt_chat_completion(messages=messages, model='gpt-3.5-turbo-0125', temperature=1.0)
            print(response)
            question_list[i]['question'] = response
    return question_list

def read_entity2id(file_path):
    entity_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  
                entity, entity_id = line.split('\t')
                entity_dict[entity_id] = entity
    return entity_dict

def main():
    triple_list = []
    with open(args.triplet_path, 'r', encoding='utf-8') as file:
        for line in file:
            triplets = line.strip().replace("_", " ").split('\t')
            triple_list.append(triplets)
    with open(args.question_path, 'r') as file:
        question_json = json.load(file)
    question_list = [q['question'] for q in question_json]
    # #fact retrival
    background_list = retrieve(args.d,'all-mpnet-base-v2', question_list, triple_list)
    rewrite
    rewrite_question = rewrite(background_list,question_list)

    with open(args.rewrite_output_path, 'w', encoding='utf-8') as file:
        json.dump(rewrite_question, file, ensure_ascii=False, indent=4)
    #time retrival
    fact_list = retrieve(args.d,args.retrieve_name, question_list, triple_list) 
    # generate prompt
    assert len(question_list)==len(fact_list)
    result_list = []
    
    for i in range(len(fact_list)):
        question = question_json[i]['question']
        fact = fact_list[i]['fact']
        print(fact)
        triple_id = fact_list[i]['triple']
        sentences = []
        answers = question_json[i]['answer']
        text = f"Based on the facts, please answer the given question. Keep the answer as simple as possible and return all the possible answers as a list.\n Facts:{fact}\nQuestion:\n {question}?" 
        while len(text) > 1024:
            fact.pop()   
            triple_id.pop()
            text = f"Based on the facts, please answer the given question. Keep the answer as simple as possible and return all the possible answers as a list.\n Facts:{fact}\nQuestion:\n {question}?"   

        if args.type == "train":
            formatted_data = {
                "instruction":text,
                "output":str(answers),
                "input":"",
                "embedding_ids":triple_id
            }
        else:
            formatted_data = {
                "text": text,
                "answers": str(answers),
                "question":question,
                "embedding_ids":triple_id
                }
        result_list.append(formatted_data)

    with open(args.output_path, "w",encoding='utf-8') as json_file:
        json.dump(result_list, json_file, indent=4)  # 使用缩进格式化 JSON 数据


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--triplet_path", type=str)
    argparser.add_argument("--question_path", type=str)
    argparser.add_argument("--rewrite_output_path", type=str)
    argparser.add_argument("--retrieve_name", type=str)
    argparser.add_argument("--output_path", type=str)
    argparser.add_argument("--d",default='time')
    argparser.add_argument("--type",default='train')

    args = argparser.parse_args()
    main()