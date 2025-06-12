import torch

from embedding import NumEmbedding, WordEmbedding
from torch import nn

# 根據不同特徵種類，用WordEmbedding或NumEmbedding轉為embedding
class FeatureProcessor(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, padding_idx=0, hidden_dropout_prob=0, layer_norm_eps=1e-5):
        super().__init__()
        self.word_embedding = WordEmbedding(vocab_size, hidden_dim, padding_idx, hidden_dropout_prob, layer_norm_eps)
        self.num_embedding = NumEmbedding(self.word_embedding, hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
    def forward(self, x_num=None, num_col_input_ids=None, num_att_mask=None, x_cat_input_ids=None, cat_att_mask=None, x_bin_input_ids=None, bin_att_mask=None):
        embeddings = []
        attention_masks = []

        if x_num is not None and num_col_input_ids is not None:
            num_feature_embedding = self.num_embedding(x_num, num_col_input_ids, num_att_mask)
            num_feature_embedding = self.align_layer(num_feature_embedding)
            num_attention_mask = torch.ones(num_feature_embedding.size(0), num_feature_embedding.size(1), device=num_feature_embedding.device)
            embeddings.append(num_feature_embedding)
            attention_masks.append(num_attention_mask)

        if x_cat_input_ids is not None:
            cat_feature_embedding = self.word_embedding(x_cat_input_ids)
            cat_feature_embedding = self.align_layer(cat_feature_embedding)
            embeddings.append(cat_feature_embedding)
            attention_masks.append(cat_att_mask)

        if x_bin_input_ids is not None:
            if x_bin_input_ids.size(1) == 0: # all false, pad zero
                x_bin_input_ids = torch.zeros(x_bin_input_ids.size(0), dtype=int, device=x_bin_input_ids.device).unsqueeze(-1)
            bin_feature_embedding = self.word_embedding(x_bin_input_ids)
            bin_feature_embedding = self.align_layer(bin_feature_embedding)
            embeddings.append(bin_feature_embedding)
            attention_masks.append(bin_att_mask)

        embeddings = torch.cat(embeddings, dim=1)
        attention_masks = torch.cat(attention_masks, dim=1)

        return {
            'embedding': embeddings,
            'attention_mask': attention_masks
        }

if __name__ == '__main__':
    from feature_extractor import FeatureExtractor
    from load_data import load_data

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../playground-series-s5e6/train.csv')

    x = valid_dataset[0]

    feature_extractor = FeatureExtractor(categorical_features, numerical_features)
    encoded_inputs = feature_extractor(x)

    feature_processor = FeatureProcessor(feature_extractor.vocab_size)
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
