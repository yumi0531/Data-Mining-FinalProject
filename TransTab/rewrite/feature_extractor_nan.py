import torch

from transformers import BertTokenizerFast

# 將特徵及類別轉換為token
class NANFeatureExtractor:
    def __init__(self, categorical_features, numerical_features, binary_features=[]):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('../transtab/tokenizer')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.binary_features = binary_features

    def __call__(self, x, device):
        x = x.copy()

        if len(self.categorical_features) > 0:
            x_cat = x[self.categorical_features]
            x_cat = x_cat.fillna('')
            x_cat = x_cat.astype(str)
            x[self.categorical_features] = x_cat

        if len(self.numerical_features) > 0:
            x_num = x[self.numerical_features]
            x_num = x_num.fillna(0)
            x_num = x_num.round(2)
            x_num = x_num.astype(str)
            x[self.numerical_features] = x_num
        
        x = x.apply(lambda x: x.name + ' is ' + x)
        x_str = x.agg(' '.join, axis=1)
        x_str = x_str.values.tolist()
        x_ts = self.tokenizer(x_str, padding=True, add_special_tokens=False, return_tensors='pt')
        
        input_ids = x_ts['input_ids']
        attention_mask = x_ts['attention_mask']
        na_mask = input_ids == self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'na_mask': na_mask.to(device)
        }

if __name__ == '__main__':
    from load_data_mush import load_data

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../playground-series-s5e6/train.csv')

    x = valid_dataset[0]
    feature_extractor = NANFeatureExtractor(categorical_features, numerical_features)
    encoded_inputs = feature_extractor(x, 'cuda')
    print('┌──────────────────┐')
    print('│ FeatureExtractor │')
    print('└──────────────────┘')
    print('{')
    for key, value in encoded_inputs.items():
        print(f'\'{key}\':')
        print(f'{value},')
    print('}')
