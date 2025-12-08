# **P14 - Transparent Minds (NLP Project)**

A visual exploration tool for interpreting encoder-only transformer language models.
The interface lets you run text through multiple NLP models and inspect attention, saliency, and PCA-based projections through a set of interactive visualizations.

---

## **Setup**

```bash
conda env create -f nlp_proj.yml
conda activate nlp_proj
```
---

## **Usage**

Run gui.py or the notebook (might not be implemented yet).

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
<img width="1774" height="1100" alt="grafik" src="https://github.com/user-attachments/assets/ab149a7b-7a84-4390-96b6-69638189ce00" />

---

### **Attention Heatmap**

Displays the attention distribution across layers and heads.
Layer/head selection can be adjusted interactively.
<img width="1735" height="897" alt="grafik" src="https://github.com/user-attachments/assets/e38939af-f7e4-4527-a1b3-79a0d5bd576d" />

---

### **Attention Line Graph**

A complementary representation of token attention:

* **Blue lines**: outgoing attention *from* the selected token
* **Orange lines**: incoming attention *to* the selected token

Hovering over tokens provides a clearer sense of directional relationships.
<img width="1730" height="907" alt="grafik" src="https://github.com/user-attachments/assets/934b582d-fd2a-4683-a5a7-0e1cd57f4309" />

---

### **Token Influence Graph**

Shows each token’s saliency score.
Useful for identifying which words contribute most strongly to the model’s output.
<img width="1300" height="662" alt="grafik" src="https://github.com/user-attachments/assets/71c88747-6363-4918-b0c6-a78a2c1d97b6" />

---

### **Saliency Timeline**

Displays token saliency across embedding/layer depths as a heatmap, highlighting how influence evolves through the network.
<img width="1420" height="766" alt="grafik" src="https://github.com/user-attachments/assets/db449468-6b90-43a9-88a6-58fbb2145f61" />

---

### **Saliency Projection (PCA)**

Uses PCA to reduce saliency dimensions and visualize relationships between tokens.
Correlated or similarly influential words tend to cluster together in the scatter plot.
<img width="1559" height="774" alt="grafik" src="https://github.com/user-attachments/assets/28f87626-69e7-45db-9b29-1eca2ec50c2b" />

---

### **Hidden State Evolution**

Shows the evolution of the L2 Norm of the Hiddenstates for each token.
<img width="1397" height="772" alt="grafik" src="https://github.com/user-attachments/assets/5cb4d09e-b276-4cc0-8f5d-04ad7b22ffa6" />

---

### **Integrated Gradients at Hidden States**

Each tokens Integrated Gradients at selectable Hidden States.
<img width="1562" height="690" alt="grafik" src="https://github.com/user-attachments/assets/005e86fd-68ee-4162-ad1d-13abcfb0eefc" />

---

### **Attention Rollout**

Attention Rollout animated over the layers.
<img width="865" height="709" alt="grafik" src="https://github.com/user-attachments/assets/0c1a4877-a0e2-48f8-98bc-b7bcd85bd773" />

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
