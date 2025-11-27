#pip install transformers torch

from transformers import RobertaTokenizer, RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as ModelOutputs
import torch

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True, output_attentions=True)

def get_model_outputs(text: str) -> ModelOutputs:
    inputs = tokenizer(text, return_tensors="pt")
    outputs: ModelOutputs = model(**inputs)
    return outputs