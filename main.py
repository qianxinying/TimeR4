import os
import csv
import pickle
import time
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
from retrival import Retrieval

# Set OpenAI API credentials from environment variables.
# It's recommended to use environment variables for sensitive data like API keys
# for security reasons, rather than hardcoding them in the script.
os.environ['OPENAI_BASE_URL'] = 'YOUR_OPENAI_URL'
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_KEY'

def retrieve(model_name, question_list, triple_list):
    """
    Retrieves relevant facts for a list of questions using a retrieval model.

    Args:
        d (str): A parameter for the Retrieval class, its specific function is not detailed here.
        model_name (str): The name of the sentence transformer model to be used for encoding.
        question_list (list): A list of questions to find relevant facts for.
        triple_list (list): A list of triples (subject, predicate, object) representing the knowledge base.

    Returns:
        list: A list of the most relevant facts for each question.
    """
    retriever = Retrieval( model_name, question_list, triple_list,None,None)
    # Compute similarity between questions and the corpus of triples.
    # n=15 specifies that the top 15 most similar facts should be returned.
    distances, corpus_ids = retriever.compute_similarity(n=15)
    # Retrieve the actual fact strings based on the computed similarities.
    # The re_rank parameter is set to False, meaning no additional re-ranking is performed.
    fact_list = retriever.get_result(distances, corpus_ids, question_list, re_rank=False)
    return fact_list

def gpt_chat_completion(**kwargs):
    """
    Sends a request to the OpenAI chat completion API with exponential backoff for retries.

    Args:
        **kwargs: Keyword arguments to be passed to the OpenAI client's
                  chat.completions.create method. This typically includes
                  'model', 'messages', 'temperature', etc.

    Returns:
        str: The content of the response message from the API.
    """
    backoff_time = 1  # Initial delay in seconds before retrying.
    while True:
        try:
            client = OpenAI()
            response = client.chat.completions.create(**kwargs)
            # Handle different response types.
            if isinstance(response, str):
                return response_parser(response)
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(e)
            # Increase the backoff time for the next retry to avoid overwhelming the server.
            time.sleep(backoff_time)
            backoff_time *= 1.5

def response_parser(response):
    """
    Parses a streaming response from the OpenAI API.

    Args:
        response (str): The raw streaming response string.

    Returns:
        str: The concatenated content from the streaming response.
    """
    # Clean and split the response into individual data chunks.
    response = response.strip().split("data: ")[1:]
    result = ''
    for r in response:
        if r == '[DONE]':
            break
        # Parse the JSON data and extract the content.
        delta = json.loads(r)['choices'][0]['delta']
        if 'content' in delta:
            result += delta['content']
    return result

def rewrite(fact_list, question_list):
    """
    Rewrites questions by incorporating information from retrieved facts.

    Args:
        fact_list (list): A list of facts corresponding to each question.
        question_list (list): A list of question dictionaries.

    Returns:
        list: The list of question dictionaries with updated 'question' fields.
    """
    # Iterate through each question and its corresponding fact.
    for i in tqdm(range(len(question_list)), desc="Processing data", unit="i"):
        question = question_list[i]['question']
        fact = fact_list[i][0]
        # Construct a prompt for the GPT model to rewrite the question.
        prompt = f"Replace facts in questions with explicit information from provided facts without any explanation.If you are not sure about the answer, output the original question. For instance, from the fact \"Herman Van Rompuy replaced by Donald Tusk from 2009 to 2014.\" modify the question \"what was donald tusk's position before he was replaced by herman van rompuy?\" to \what was donald tusk's position before 2009?\" Here is your turn: Question: {question} Fact:{fact}."
        print(question)
        messages = [{"role": "user", "content": prompt}]
        # Get the rewritten question from the GPT model.
        response = gpt_chat_completion(messages=messages, model='gpt-3.5-turbo-0125', temperature=1.0)
        print(response)
        # Update the question in the list.
        question_list[i]['question'] = response
    return question_list

def read_entity2id(file_path):
    """
    Reads an entity-to-ID mapping file.

    Args:
        file_path (str): The path to the file containing entity-to-ID mappings.

    Returns:
        dict: A dictionary mapping entity IDs to entity names.
    """
    entity_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entity, entity_id = line.split('\t')
                entity_dict[entity_id] = entity
    return entity_dict

def main():
    """
    Main function to execute the question rewriting and data generation pipeline.
    """
    # Load the knowledge base triples.
    triple_list = []
    with open(args.fact_path, 'r', encoding='utf-8') as file:
        for line in file:
            triplets = line.strip().replace("_", " ").split('\t')
            triple_list.append(triplets)

    # Load the questions.
    with open(args.question_path, 'r') as file:
        question_json = json.load(file)
    question_list = [q['question'] for q in question_json]

    # Retrieve background information for rewriting.
    background_list = retrieve(args.retrieve_name, question_list, triple_list)

    # Rewrite the questions based on the retrieved background information.
    rewrite_question = rewrite(background_list, question_json)

    # Save the rewritten questions.
    with open(args.rewrite_output_path, 'w', encoding='utf-8') as file:
        json.dump(rewrite_question, file, ensure_ascii=False, indent=4)

    # Retrieve facts for the final prompt generation.
    fact_list = retrieve(args.retrieve_name, question_list, triple_list)

    # Generate prompts for the final model.
    assert len(question_list) == len(fact_list)
    result_list = []

    for i in range(len(fact_list)):
        question = question_json[i]['question']
        fact = fact_list[i]['fact']
        print(fact)
        triple_id = fact_list[i]['triple']
        answers = question_json[i]['answer']

        # Construct the prompt text.
        text = f"Based on the facts, please answer the given question. Keep the answer as simple as possible and return all the possible answers as a list.\n Facts:{fact}\nQuestion:\n {question}?"

        # Format the data based on whether it's for training or another purpose.
        if args.type == "train":
            formatted_data = {
                "instruction": text,
                "output": str(answers),
                "input": "",
                "embedding_ids": triple_id
            }
        else:
            formatted_data = {
                "text": text,
                "answers": str(answers),
                "question": question,
                "embedding_ids": triple_id
            }
        result_list.append(formatted_data)

    # Save the final formatted data.
    with open(args.output_path, "w", encoding='utf-8') as json_file:
        json.dump(result_list, json_file, indent=4)

if __name__ == "__main__":
    # Set up the argument parser to handle command-line arguments.
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--fact_path", type=str, default='timeR4/datasets/MultiTQ/kg/full.txt', help="Path to the knowledge base fact file.")
    argparser.add_argument("--question_path", type=str, help="Path to the question file.")
    argparser.add_argument("--rewrite_output_path", type=str, help="Path to save the rewritten questions.")
    argparser.add_argument("--retrieve_name", type=str, help="Name of the retrieval model.",default='sentence-transformers/all-mpnet-base-v2')
    argparser.add_argument("--output_path", type=str, help="Path to save the final output.")
    argparser.add_argument("--type", default='train', help="Type of data generation (e.g., 'train').")

    # Parse the command-line arguments.
    args = argparser.parse_args()
    main()