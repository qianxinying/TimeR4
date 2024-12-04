from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from torch.utils.data import DataLoader
import json
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, LoggingHandler, models, util, InputExample
import argparse
from sentence_transformers import losses
def main(args):

  with open(args.path, 'r') as file:
      examples = json.load(file)

  train_size = int(len(examples) * 0.8)
  eval_size = len(examples) - train_size
  # Define your train examples.
  train_data = []
  for example in examples[:train_size]:
    train_data.append(InputExample(texts=[example['question'], example['positive']], label=1.0))
    for negative in example['negative']:   
      train_data.append(InputExample(texts=[example['question'],negative], label=0.0))

  # Define your evaluation examples
  sentences1 = []
  sentences2 = []
  scores = []
  for example in examples[train_size:]:   
    for i in range(5):
      sentences1.append(example['question'])
    sentences2.append(example['positive'])
    for neg in example['negative']:
      sentences2.append(neg)
    scores.extend([1,0,0,0,0])
    assert len(sentences1)==len(sentences2)

  #Define the model. Either from scratch of by loading a pre-trained model
  word_embedding_model = models.Transformer('/home/qxy/all-mpnet-base-v2', max_seq_length=128)

  tokens = ["[B_SEP]", "[A_SEP]","[F_SEP]",'[L_SEP]']
  for i in range(4017):
    time_token = f'[Time_{i}]'
    tokens.append(time_token)
  word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
  word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))


  pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
  model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

  evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
  # Define your train dataset, the dataloader and the train loss
  train_dataset = SentencesDataset(train_data, model)
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=256)
  # train_loss = losses.CosineSimilarityLoss(model)
  train_loss = losses.ContrastiveTensionLossInBatchNegatives(model, scale=1, similarity_fct=util.dot_score)


  # Tune the model1``
  model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=1000, evaluator=evaluator, evaluation_steps=1000, optimizer_params={"lr": 5e-5},output_path=args.model_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", type=str)
    argparser.add_argument("--model_name", type=str)

    args = argparser.parse_args()
    main(args)