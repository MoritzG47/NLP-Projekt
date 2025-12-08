from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as ModelOutputs

class LanguageModel:
    def __init__(self):
        MODEL = "distilbert-base-uncased" # distilbert-base-uncased, roberta-base, bert-base-uncased

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL, output_hidden_states=True, output_attentions=True)
        self.model.eval()

    def get_model_outputs(self, text: str) -> tuple[ModelOutputs, list[str]]:
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        outputs: ModelOutputs = self.model(**inputs)
        saliency = self.token_influence(inputs)
        return outputs, tokens, saliency

    def change_model(self, new_model: str):
        if new_model in ["distilbert-base-uncased", "roberta-base", "bert-base-uncased"]:
            self.tokenizer = AutoTokenizer.from_pretrained(new_model)
            self.model = AutoModel.from_pretrained(new_model, output_hidden_states=True, output_attentions=True)
            self.model.eval()

    def token_influence(self, input) -> list[float]:
        input_ids = input["input_ids"]

        embedding_layer = self.model.embeddings.word_embeddings
        inputs_embeds = embedding_layer(input_ids)
        inputs_embeds.requires_grad_(True)
        outputs = self.model(inputs_embeds=inputs_embeds)
        last_hidden_state = outputs.last_hidden_state
        # Use CLS embedding as "score"
        cls_score = last_hidden_state[:, 0, :].sum()  # scalar needed for backprop

        # Reset gradients to zero before backpropagation
        self.model.zero_grad()
        cls_score.backward()

        # Retrieve embeddings & grads
        emb_layer = self.model.embeddings.word_embeddings
        embeddings = emb_layer(input_ids)
        grads = input_ids.grad

        grad_wrt_weights = emb_layer.weight.grad[input_ids]
        # saliency = gradient × input
        saliency = (embeddings * grad_wrt_weights)
        return saliency

if __name__ == "__main__":
    lm = LanguageModel()
