#pip install transformers torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as ModelOutputs
#import torch

MODEL = "bert"

if MODEL == "roberta":
    from transformers import RobertaTokenizer, RobertaModel
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True, output_attentions=True)
elif MODEL == "bert":
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)

def get_model_outputs(text: str) -> tuple[ModelOutputs, list[str]]:
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    outputs: ModelOutputs = model(**inputs)
    return outputs, tokens

