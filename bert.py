from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained("../data/bert-base-uncased/bert-base-uncased-vocab.txt")
bert = BertModel.from_pretrained("../data/bert-base-uncased/")

s = "I'm not sure, this can work, lol -.-"
tokens = tokenizer.tokenize(s)
print("\\".join(tokens))
ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
print(ids.shape)
result = bert(ids, output_all_encoded_layers=True)
print(result)

