import torch

from embedding import WordEmbedding
from torch import nn

# 根據不同特徵種類，用WordEmbedding或NumEmbedding轉為embedding
class NANFeatureProcessor(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, padding_idx=0, hidden_dropout_prob=0, layer_norm_eps=1e-5):
        super().__init__()
        self.word_embedding = WordEmbedding(vocab_size, hidden_dim, padding_idx, hidden_dropout_prob, layer_norm_eps)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
    def forward(self, input_ids=None, attention_mask=None, na_mask=None):
        embeddings = self.word_embedding(input_ids)  
        embeddings = self.align_layer(embeddings)
        embeddings[na_mask] = 0
        attention_mask[na_mask] = 0
        return {
            'embedding': embeddings,
            'attention_mask': attention_mask
        }

if __name__ == '__main__':
    from feature_extractor_nan import NANFeatureExtractor
    from load_data_mush import load_data

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../playground-series-s5e6/train.csv')

    x = valid_dataset[0]

    feature_extractor = NANFeatureExtractor(categorical_features, numerical_features)
    encoded_inputs = feature_extractor(x, 'cuda')

    feature_processor = NANFeatureProcessor(feature_extractor.vocab_size).cuda()
    processed_result = feature_processor(**encoded_inputs)
    embeddings = processed_result['embedding']
    attention_masks = processed_result['attention_mask']

    print('┌────────────┐')
    print('│ embeddings │')
    print('└────────────┘')
    print(f'shape: {tuple(embeddings.shape)}')
    print()
    print('┌─────────────────┐')
    print('│ attention_masks │')
    print('└─────────────────┘')
    print(f'shape: {tuple(attention_masks.shape)}')
