import torch

from transformers import CLIPTokenizer, CLIPTextModel

class CLIPFeatureExtractor:
    def __init__(self, categorical_features, numerical_features, binary_features=[]):
        self.tokenizer = CLIPTokenizer.from_pretrained('zer0int/LongCLIP-SAE-ViT-L-14')
        self.text_model = CLIPTextModel.from_pretrained('zer0int/LongCLIP-SAE-ViT-L-14').eval()
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.binary_features = binary_features
        self.nan_token_id = self.tokenizer('nan')['input_ids'][1]
    def __call__(self, x, device):
        self.text_model.to(device)
        x = x.copy()

        x[self.numerical_features] = x[self.numerical_features].round(2)
        x = x.astype(str)
        x = x.apply(lambda x: x.name + ' ' + x)
        x = x.agg(' '.join, axis=1)
        x = x.values.tolist()

        tokens = self.tokenizer(x, padding=True, return_tensors='pt').to(device)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        na_mask = input_ids == self.nan_token_id

        attention_mask[na_mask] = 0

        with torch.no_grad():
            embeddings = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,1:,:]

        na_mask = na_mask[:,1:]
        embeddings[na_mask] = 0

        return {
            'embedding': embeddings,
            'attention_mask': attention_mask[:,1:]
        }

if __name__ == '__main__':
    from load_data import load_data

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../MushroomDataset/secondary_data.csv')

    data = valid_dataset[0]

    feature_extractor = CLIPFeatureExtractor(categorical_features, numerical_features)

    encoded_inputs = feature_extractor(data.head(256), 'cuda')

    print('┌──────────────────┐')
    print('│ FeatureExtractor │')
    print('└──────────────────┘')
    print('{')
    for key, value in encoded_inputs.items():
        print(f'\'{key}\':')
        print(f'{value},')
    print('}')
