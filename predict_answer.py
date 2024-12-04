import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset
import json
from multiprocessing import Pool
from functools import partial
import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score
import ast

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s)
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_hit(prediction, answer):
    answer = ast.literal_eval(answer)
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_result(predict_file, cal_f1=True, topk = -1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = "detailed_eval_result_top_{topk}.jsonl" if topk > 0 else 'detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            # id = data['id']
            prediction = data['prediction']
            answer = data['ground_truth']
            if cal_f1:
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n")
                else:
                    prediction = extract_topk_prediction(prediction, topk)
  
                prediction_str = ' '.join(prediction)
                hit = eval_hit(prediction_str, answer)
                hit_list.append(hit)
                f2.write(json.dumps({'prediction': prediction, 'ground_truth': answer, 'hit': hit}) + '\n')
            else:
                hit = eval_hit(prediction, answer)
                hit_list.append(hit)
                f2.write(json.dumps({'prediction': prediction, 'ground_truth': answer,'hit': hit}) + '\n')
    

    result_str = " Hit: " + str(sum(hit_list) * 100 / len(hit_list))
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else 'eval_result.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)
    with open(eval_result_path, 'w') as f:
        f.write(result_str)

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout
    else:
        with open(path, "r") as f:
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
        fout = open(path, "a")
        return fout



def prediction(data, model):
    question = data["text"]
    answer = data["answers"]
    prediction = model.generate_sentence(question)
    if prediction is None:
        return None
    result = {
        "question": question,
        "prediction": prediction,
        "ground_truth": answer
    }
    return result


def main(args,LLM):
    rule_postfix = "no_rule"

    with open(args.d, "r") as json_file:
        dataset = json.load(json_file)
    print("Load dataset from finished")

    output_dir = args.predict_path
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        print("Prepare pipline for inference...")
        model.prepare_for_inference()

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout = get_output_file(output_file, force=args.force)

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction(data, model)
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    fout.close()

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--d", "-d", type=str, default="multitq")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results")
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="llama",
    )
    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()
    main(args, LLM)
