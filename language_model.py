import torch
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
    
    def token_influence(self, input) -> torch.Tensor:
        input_ids = input["input_ids"]

        embedding_layer = self.model.embeddings.word_embeddings
        inputs_embeds = embedding_layer(input_ids)
        inputs_embeds.requires_grad_(True)
        inputs_embeds.retain_grad()

        outputs = self.model(inputs_embeds=inputs_embeds)
        cls_score = outputs.last_hidden_state[:, 0, :].sum()

        self.model.zero_grad()
        cls_score.backward()

        saliency = (inputs_embeds.grad * inputs_embeds)
        return saliency


if __name__ == "__main__":
    lm = LanguageModel()
