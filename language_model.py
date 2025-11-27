#pip install transformers torch

from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True, output_attentions=True)

text = "RoBERTa models are powerful for feature extraction."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

hidden_states = outputs.hidden_states
attentions = outputs.attentions

