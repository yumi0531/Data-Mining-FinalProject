import math
import torch

from torch import nn

# 文字embedding
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, padding_idx=0, hidden_dropout_prob=0, layer_norm_eps=1e-5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim, padding_idx),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            nn.Dropout(hidden_dropout_prob)
        )
        nn.init.kaiming_normal_(self.net[0].weight) # embedding layer
    def forward(self, x):
        return self.net(x)

# 文字embedding乘上數值
class NumEmbedding(nn.Module):
    def __init__(self, word_embedding, hidden_dim=128):
        super().__init__()
        self.word_embedding = word_embedding
        a = - 1 / math.sqrt(hidden_dim)
        b = 1 / math.sqrt(hidden_dim)
        self.bias = nn.Parameter(torch.empty(1, 1, hidden_dim).uniform_(a, b))
    def forward(self, num, num_feature_ids, num_attention_mask):
        batch_size = num.size(0)
        feature_embedding = self.word_embedding(num_feature_ids)
        feature_embedding = feature_embedding * num_attention_mask.unsqueeze(-1)
        feature_embedding = feature_embedding.sum(1) / num_attention_mask.sum(1, keepdim=True)
        feature_embedding = feature_embedding.unsqueeze(0).expand((batch_size, -1, -1))
        return feature_embedding * num.unsqueeze(-1).float() + self.bias

if __name__ == '__main__':
    import sys
    sys.path.append('..')

    import transtab

    from feature_extractor import FeatureExtractor

    transtab.random_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, binary_features = transtab.load_data('../dataset')
    print('########################################')

    x = valid_dataset[0]

    feature_extractor = FeatureExtractor(categorical_features, numerical_features, binary_features)
    encoded_inputs = feature_extractor(x)
    cat_input_ids = encoded_inputs['x_cat_input_ids']

    word_embedding = WordEmbedding(feature_extractor.vocab_size)
    cat_data_embedding = word_embedding(cat_input_ids)
    print(f'{tuple(cat_input_ids.shape)} --{word_embedding.__class__.__name__}-> {tuple(cat_data_embedding.shape)}')

    x_num_ts = encoded_inputs['x_num']
    num_input_ids = encoded_inputs['num_col_input_ids']
    num_attention_mask = encoded_inputs['num_att_mask']

    num_embedding = NumEmbedding(word_embedding)
    num_data_embedding = num_embedding(x_num_ts, num_input_ids, num_attention_mask)
    print(f'{tuple(x_num_ts.shape)} + {tuple(num_input_ids.shape)} + {tuple(num_attention_mask.shape)} --{num_embedding.__class__.__name__}-> {tuple(num_data_embedding.shape)}')
