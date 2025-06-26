# LongMaskTab

**Modeling Missing Values and Long Feature Interactions in Tabular Transformers**  
[Team_9 Final Report (PDF)](https://github.com/user-attachments/files/20918235/Team_9.pdf)

## Overview

**LongMaskTab** is an enhanced variant of the TransTab framework designed for robust tabular data modeling. It addresses two major limitations of traditional tabular transformers:

1. **Missing Value Handling**: Inspired by the NAIM strategy, we handle missing data using `[MASK]` tokens, and mask their embeddings and attention to prevent them from affecting the model.
2. **Long Feature Interaction Modeling**: By replacing the default BERT tokenizer with a Long-CLIP tokenizer, the model supports input sequences up to 248 tokens, enabling richer semantic feature representations.

Our experiments on the Mushroom dataset show that LongMaskTab outperforms the original TransTab, especially under high missingness.



## Key Features

- We improve the original TransTab model by integrating a built-in mechanism for handling missing values, where missing fields are directly dropped, enabling effective learning from datasets with substantial missingness.
- We use NAIM (Not Another Imputation Method), which applies masking at the embedding layer, eliminating the need for external pre-imputation strategies.
- We evaluate our method on the Mushroom dataset, where some columns have up to 85% missing values and nearly every feature has around 20% missingness. Without generating synthetic data, our approach still contributes a measurable improvement over the original TransTab and decision
tree-based imputation methods.
- Considering the large number of features in tabular data, we adopt LongCLIP to support extended token sequences (up to 248 tokens), allowing richer semantic representations.
- We further investigate the impact of different natural language concatenation strategies for textual column representations, highlighting how design choices in prompt construction affect model performance.



## Project Structure
LongMaskTab/  
├── feature_extractor_CLIP.py      # CLIP-based tokenization for tabular data & Masking logic for missing values  
├── feature_processor_CLIP.py        
├── train.py                       # Training script  
├── evaluate.py                    # Evaluation script  
├── model_CLIP.py                  # Transformer model architecture   
├── training_config.yml            # Configurable parameters   



## Dataset

We use the [Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset) containing:
- 61,000+ samples
- 20 features (both categorical and numerical)
- Binary classification: edible (e) or poisonous (p)

Missing values were randomly introduced up to 50% for robustness testing.



## Getting Started


### 1. Install Dependencies & Dataset
Download the [Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset) and place the extracted CSV file in the same directory level as the `LongMaskTab` folder.

```bash
pip install -r requirements.txt
```


### 2. Train model
edit model hyperparamter on "training_config.yml" file.
```bash
python train.py
```

### 3. Evaluate the Model
```bash
python evaluate.py
```


---
This project was developed as part of the Data Mining Final Project at National Cheng Kung University (NCKU), Spring 2025.
