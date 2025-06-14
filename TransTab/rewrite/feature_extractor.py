import torch

from transformers import BertTokenizerFast

# 將特徵及類別轉換為token
class FeatureExtractor:
    def __init__(self, categorical_features, numerical_features, binary_features=[]):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('../transtab/tokenizer')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.binary_features = binary_features

    def __call__(self, x, device):
        encoded_inputs = {}

        if len(self.categorical_features) > 0:
            x_cat = x[self.categorical_features].astype(str)
            x_mask = (~x_cat.isna()).astype(int)
            x_cat = x_cat.fillna('')
            x_cat = x_cat.apply(lambda x: x.name + ' ' + x) * x_mask
            x_cat_str = x_cat.agg(' '.join, axis=1)
            x_cat_str = x_cat_str.values.tolist()
            x_cat_ts = self.tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

            encoded_inputs['x_cat_input_ids'] = x_cat_ts['input_ids'].to(device) # categorical feature + value tokenized ids
            encoded_inputs['cat_att_mask'] = x_cat_ts['attention_mask'].to(device) # categorical attention mask
        
        if len(self.numerical_features) > 0:
            x_num = x[self.numerical_features]
            x_num = x_num.fillna(0)
            num_col_ts = self.tokenizer(self.numerical_features, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

            encoded_inputs['x_num'] = torch.tensor(x_num.values, dtype=float, device=device) # numerical features value
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids'].to(device) # numerical feature tokenized ids
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask'].to(device) # numerical attention mask
        
        if len(self.binary_features) > 0:
            # binary no process NaN
            x_bin = x[self.binary_features]
            x_bin_str = x_bin.apply(lambda x: x.name) * x_bin
            x_bin_str = x_bin_str.agg(' '.join, axis=1)
            x_bin_str = x_bin_str.values.tolist()
            x_bin_ts = self.tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

            if x_bin_ts['input_ids'].shape[1] > 0: # not all false
                encoded_inputs['x_bin_input_ids'] = x_bin_ts['input_ids'].to(device) # binary feature tokenized ids
                encoded_inputs['bin_att_mask'] = x_bin_ts['attention_mask'].to(device) # binary attention mask

        return encoded_inputs

if __name__ == '__main__':
    from load_data import load_data

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../playground-series-s5e6/train.csv')

    tokenizer = BertTokenizerFast.from_pretrained('../transtab/tokenizer')

    x = valid_dataset[0]
    x['High'] = (x['Temperature'] > 0.5).astype(int)
    binary_features = ['High']

    x_cat = x[categorical_features].astype(str)
    x_mask = (~x_cat.isna()).astype(int)
    x_cat = x_cat.fillna('')
    x_cat = x_cat.apply(lambda x: f'{x.name} ' + x) * x_mask
    x_cat_str = x_cat.agg(' '.join, axis=1)

    print('┌───────────┐')
    print('│ x_cat_str │')
    print('└───────────┘')
    print(x_cat_str)

    x_cat_str = x_cat_str.values.tolist()
    x_cat_ts = tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

    print()
    print('┌─────────────────┐')
    print('│ x_cat_input_ids │')
    print('└─────────────────┘')
    print(x_cat_ts['input_ids'])
    print(f'shape: {tuple(x_cat_ts["input_ids"].shape)}')

    print()
    print('┌──────────────────────────────────────┐')
    print('│ tokenizer.decode(x_cat_input_ids[0]) │')
    print('└──────────────────────────────────────┘')
    print(tokenizer.decode(x_cat_ts['input_ids'][0]))

    print()
    print('┌──────────────┐')
    print('│ cat_att_mask │')
    print('└──────────────┘')
    print(x_cat_ts['attention_mask'])
    print(f'shape: {tuple(x_cat_ts["attention_mask"].shape)}')

    print('════════════════════════════════════════')

    num_col_ts = tokenizer(numerical_features, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

    print('┌───────────────────┐')
    print('│ num_col_input_ids │')
    print('└───────────────────┘')
    print(num_col_ts['input_ids'])
    print(f'shape: {tuple(num_col_ts["input_ids"].shape)}')

    print()
    print('┌─────────────────────────────────────┐')
    print('│ tokenizer.decode(num_col_input_ids) │')
    print('└─────────────────────────────────────┘')
    for input_id in num_col_ts['input_ids']:
        print(tokenizer.decode(input_id))

    print()
    print('┌──────────────┐')
    print('│ num_att_mask │')
    print('└──────────────┘')
    print(num_col_ts['attention_mask'])
    print(f'shape: {tuple(num_col_ts["attention_mask"].shape)}')

    print('════════════════════════════════════════')

    x_bin = x[binary_features]
    x_bin_str = x_bin.apply(lambda x: x.name) * x_bin
    x_bin_str = x_bin_str.agg(' '.join, axis=1)
    
    print('┌───────────┐')
    print('│ x_bin_str │')
    print('└───────────┘')
    print(x_bin_str)

    x_bin_str = x_bin_str.values.tolist()
    x_bin_ts = tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
    
    print()
    print('┌─────────────────┐')
    print('│ x_bin_input_ids │')
    print('└─────────────────┘')
    print(x_bin_ts['input_ids'])
    print(f'shape: {tuple(x_bin_ts["input_ids"].shape)}')

    print()
    print('┌──────────────────────────────────────────────┐')
    print('│ tokenizer.decode(x_bin_input_ids=False/True) │')
    print('└──────────────────────────────────────────────┘')
    print(f'False: {tokenizer.decode(x_bin_ts["input_ids"][0])}')
    print(f'True: {tokenizer.decode(x_bin_ts["input_ids"][1])}')

    print()
    print('┌──────────────┐')
    print('│ cat_att_mask │')
    print('└──────────────┘')
    print(x_bin_ts['attention_mask'])
    print(f'shape: {tuple(x_bin_ts["attention_mask"].shape)}')

    print('════════════════════════════════════════')

    feature_extractor = FeatureExtractor(categorical_features, numerical_features, binary_features)
    encoded_inputs = feature_extractor(x)
    print('┌──────────────────┐')
    print('│ FeatureExtractor │')
    print('└──────────────────┘')
    print('{')
    for key, value in encoded_inputs.items():
        print(f'\'{key}\':')
        print(f'{value},')
    print('}')
