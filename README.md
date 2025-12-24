# DistilBERT Emotion Classification (Fine-tuned Model)

## Overview
This repository contains a fine-tuned DistilBERT model for emotion classification. The model predicts one of six emotions from text:  

- Sadness  
- Joy  
- Love  
- Anger  
- Fear  
- Surprise  

The model was trained on the Hugging Face “emotion” dataset and achieves approximately 93% accuracy on the test set.  

---

## Features
- Pre-trained DistilBERT fine-tuned for emotion detection  
- Supports inference on custom text inputs  
- Easily loadable from Hugging Face Hub or GitHub  
- Includes all tokenizer files and model weights (model.safetensors)  

---

## Getting Started

### Clone the Repository
Clone the repo and navigate to it.  

### Install Requirements
Install Python packages: transformers, datasets, torch, scikit-learn.  

---

### Load the Model
Load the tokenizer and model from Hugging Face or GitHub and set it to evaluation mode.  

---

### Predict Emotion for a Sentence
Use the model to predict one of the six emotions for a given text input.  

---

### Evaluate on the Dataset
You can evaluate the model on the “emotion” dataset to get accuracy and weighted F1 scores.  

---

## Dataset
- Hugging Face Emotion Dataset  
- Classes: Sadness, Joy, Love, Anger, Fear, Surprise  
- Train: 16,000 samples, Validation: 2,000 samples, Test: 2,000 samples  

---

## Model Files
- model.safetensors – model weights  
- tokenizer.json, vocab.txt, special_tokens_map.json, tokenizer_config.json – tokenizer files  
- config.json – model configuration  

---

## Performance
- Test Accuracy: ~93%  
- Weighted F1 Score: ~92.7%  

**Observation:** Rare classes like `love` and `surprise` have slightly lower F1 due to fewer samples.  

---

## Future Improvements
- Add more samples for rare classes (love, surprise)  
- Use class weighting or oversampling to improve F1 for underrepresented classes  
- Fine-tune with larger or augmented datasets for higher accuracy  

---

## License
This project is open source. You are free to use and modify it.
