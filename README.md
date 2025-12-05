# **P14 – Transparent Minds (NLP Project)**

A visual exploration tool for interpreting transformer language models.
The interface lets you run text through multiple NLP models and inspect attention, saliency, and PCA-based projections through a set of interactive visualizations.

---

## **Setup**

```bash
conda env create -f nlp_env.yml
conda activate nlp_env
```
---

## **Usage**

### **Random Button**

Fetches a random sentence from the filtered **MultiNLI** dataset:
[https://huggingface.co/datasets/nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli)

The sentences are not guaranteed to be well-formed, but they’re useful for quick testing.

### **Enter Button**

Allows submitting your own sentence into the textbox for processing.

### **Tabs**

Use the tab interface to switch between the available visualizations.

---

## **Tools and Visualizations**

### **Main Interface**

The primary workspace.
You can run text through multiple transformer models and display all visualizations at once.

---

### **Attention Heatmap**

Displays the attention distribution across layers and heads.
Layer/head selection can be adjusted interactively.

---

### **Attention Line Graph**

A complementary representation of token attention:

* **Blue lines**: outgoing attention *from* the selected token
* **Orange lines**: incoming attention *to* the selected token

Hovering over tokens provides a clearer sense of directional relationships.

---

### **Token Influence Graph**

Shows each token’s saliency score.
Useful for identifying which words contribute most strongly to the model’s output.

---

### **Saliency Timeline**

Displays token saliency across embedding/layer depths as a heatmap, highlighting how influence evolves through the network.

---

### **Saliency Projection (PCA)**

Uses PCA to reduce saliency dimensions and visualize relationships between tokens.
Correlated or similarly influential words tend to cluster together in the scatter plot.

---

## **AI Disclaimer**

ChatGPT was used *partially* for:

* generating visualization ideas
* assisting with debugging
* building GUI templates
* converting mathematical formulations into code
* formatting this README

---

## **Acknowledgments**

Design inspirations:
**BertViz** – [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)

---
