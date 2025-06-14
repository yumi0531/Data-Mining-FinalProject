import torch

from torch import nn
import pandas as pd
from transformers import CLIPTokenizer, CLIPTextModel


# 將特徵及類別轉換為token
class CLIPFeatureExtractor(nn.Module):

    def __init__(self,
                 categorical_features,
                 numerical_features,
                 binary_features=[]):
        super().__init__()

        # CLIP tokenizer & text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32")

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.binary_features = binary_features
        self.device_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        device = self.device_param.device
        # change everything to natural language sentences
        sentence_list = []

        for i in range(len(x)):
            row = x.iloc[i]

            parts = []
            for col in self.categorical_features:
                val = str(row[col]) if pd.notna(row[col]) else "unknown"
                parts.append(f"{col.replace('_', ' ').lower()} is {val}")

            for col in self.numerical_features:
                val = row[col] if pd.notna(row[col]) else 0.0
                parts.append(f"{col.replace('_', ' ').lower()} is {val:.2f}")

            for col in self.binary_features:
                if row[col] == 1:
                    parts.append(f"{col.replace('_', ' ').lower()}")

            sentence = ', '.join(parts)
            sentence_list.append(sentence)

            # tokenize + encode
        tokens = self.tokenizer(sentence_list,
                                padding=True,
                                truncation=True,
                                return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = self.text_encoder(
                **tokens).last_hidden_state[:, 0, :]  # 取 [CLS] 表示

        return embedding  # shape: [batch_size, 512]


if __name__ == '__main__':
    from load_data import load_data

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data(
        '../playground-series-s5e6/train.csv')

    x = valid_dataset[0]
    x['High'] = (x['Temperature'] > 0.5).astype(int)
    binary_features = ['High']
    feature_extractor = CLIPFeatureExtractor(categorical_features,
                                             numerical_features,
                                             binary_features)
    embedding = feature_extractor(x)

    print('┌──────────────────┐')
    print('│ FeatureExtractor │')
    print('└──────────────────┘')
    print("{")
    print(f"'embedding':")
    print(embedding)
    print(f'shape: {tuple(embedding.shape)}')
    print("}")
