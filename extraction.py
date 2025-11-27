import language_model
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as ModelOutputs

def extract_hidden_states(outputs: ModelOutputs) -> tuple:
    hidden_states = outputs.hidden_states  # Tuple of (layer_count, batch_size, sequence_length, hidden_size)
    return hidden_states

def extract_attentions(outputs: ModelOutputs) -> tuple:
    attentions = outputs.attentions  # Tuple of (layer_count, batch_size, num_heads, sequence_length, sequence_length)
    return attentions

def extract_all(text: str):
    outputs: ModelOutputs = language_model.get_model_outputs(text)
    hidden_states = extract_hidden_states(outputs)
    attentions = extract_attentions(outputs)
    return hidden_states, attentions