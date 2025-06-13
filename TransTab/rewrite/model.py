import math
import torch

from feature_extractor import FeatureExtractor
from feature_processor import FeatureProcessor
from torch import nn

class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.gate = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid()
        )

        self.linear = nn.Linear(d_model, dim_feedforward)

        self.gate_out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.gate_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src_key_padding_mask = ~src_key_padding_mask.bool()
        attention = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        attention = self.dropout(attention)
        x = self.attention_norm(x + attention)

        g = self.gate(x)
        h = self.linear(x)
        g = g * h
        g = self.gate_out(g)
        x = self.gate_norm(x + g)
        return x

class GatedTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layer=2, num_attention_head=2, hidden_dropout_prob=0, ffn_dim=256):
        super().__init__()
        self.encoders = nn.ModuleList([
            GatedTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dim_feedforward=ffn_dim,
                dropout=hidden_dropout_prob,
            )
        ])
        if num_layer > 1:
            encoder_layer = GatedTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dim_feedforward=ffn_dim,
                dropout=hidden_dropout_prob,
            )
            self.encoders.append(nn.TransformerEncoder(encoder_layer, num_layer - 1, enable_nested_tensor=False))

    def forward(self, embedding, attention_mask):
        for encoder in self.encoders:
            embedding = encoder(embedding, src_key_padding_mask=attention_mask)
        return embedding

class CLSToken(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        a = - 1 / math.sqrt(hidden_dim)
        b = 1 / math.sqrt(hidden_dim)
        self.token = nn.Parameter(torch.empty(hidden_dim).uniform_(a, b))

    def forward(self, embedding, attention_mask):
        batch_size = embedding.size(0)
        token = self.token.view(1, -1).expand(batch_size, 1, -1)
        embedding = torch.cat([token, embedding], dim=1)
        token_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        attention_mask = torch.cat([token_mask, attention_mask], dim=1)
        return {
            'embedding': embedding,
            'attention_mask': attention_mask
        }

class BaseModel(nn.Module):
    def __init__(self, categorical_features=None, numerical_features=None, binary_features=None, hidden_dim=128, num_layer=2, num_attention_head=8, hidden_dropout_prob=0.1, ffn_dim=256):
        super().__init__()
        self.feature_extractor = FeatureExtractor(categorical_features, numerical_features, binary_features)
        self.feature_processor = FeatureProcessor(self.feature_extractor.vocab_size, hidden_dim, self.feature_extractor.pad_token_id, hidden_dropout_prob)

        self.cls_token = CLSToken(hidden_dim=hidden_dim)
        self.encoder = GatedTransformerEncoder(hidden_dim, num_layer, num_attention_head, hidden_dropout_prob, ffn_dim)

    def forward(self, x):
        tokenized = self.feature_extractor(x)
        embedding = self.feature_processor(**tokenized)
        embedding = self.cls_token(**embedding)
        embedding = self.encoder(**embedding)
        return embedding

class Classifier(nn.Module):
    def __init__(self, categorical_features, numerical_features, binary_features=[], num_class=2, hidden_dim=128, num_layer=2, num_attention_head=8, hidden_dropout_prob=0, ffn_dim=256):
        super().__init__()
        self.base_model = BaseModel(categorical_features, numerical_features, binary_features, hidden_dim, num_layer, num_attention_head, hidden_dropout_prob, ffn_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_class if num_class > 2 else 1)
        )

        # if num_class > 2:
        #     self.loss_fn = nn.CrossEntropyLoss()
        # else:
        #     self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        embeddings = self.base_model(x)
        embedding = embeddings[:, 0, :]
        logits = self.classifier(embedding)
        return logits

        # if self.num_class == 2:
        #     y_ts = torch.tensor(y.values).float()
        #     loss = self.loss_fn(logits.flatten(), y_ts)
        # else:
        #     y_ts = torch.tensor(y.values).long()
        #     loss = self.loss_fn(logits, y_ts)

if __name__ == '__main__':
    from load_data import load_data

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../playground-series-s5e6/train.csv')

    x = valid_dataset[0]

    model = Classifier(categorical_features, numerical_features, num_class=7).cuda()

    with torch.no_grad():
        predict = model(x)
    print(f'{tuple(x.shape)} --{model.__class__.__name__}-> {tuple(predict.shape)}')
