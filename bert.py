#coding=utf-8
import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
inputs = tokenizer("让我们试试BERT模型吧。模型是不是很厉害？", return_tensors="pt")
print(inputs)
print(tokenizer.decode(inputs["input_ids"].data.cpu().numpy().reshape(-1)))
outputs = model(**inputs)
# 词向量 句子向量
sequence_outputs, pooled_outputs = outputs
print(sequence_outputs.shape)
print(pooled_outputs.shape)