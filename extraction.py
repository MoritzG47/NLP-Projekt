import language_model
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as ModelOutputs

def extract_all(text: str):
    outputs, tokens = language_model.get_model_outputs(text)
    hidden_states = outputs.hidden_states       # Tuple of (layer_count, batch_size, sequence_length, hidden_size)
    attentions = outputs.attentions             # Tuple of (layer_count, batch_size, num_heads, sequence_length, sequence_length)
    return hidden_states, attentions, tokens