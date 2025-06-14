import torch
from torch import nn


class CLIPFeatureProcessor(nn.Module):

    def __init__(self, input_dim=512, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(nn.LayerNorm(input_dim),
                                       nn.Linear(input_dim, output_dim))

    def forward(self, embedding):
        return self.projector(embedding)


if __name__ == '__main__':
    from feature_extractor_CLIP import CLIPFeatureExtractor
    from load_data import load_data

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data(
        '../playground-series-s5e6/train.csv')

    x = valid_dataset[0]

    feature_extractor = CLIPFeatureExtractor(categorical_features,
                                             numerical_features)
    encoded_inputs = feature_extractor(x)

    processor = CLIPFeatureProcessor()
    embeddings = processor(encoded_inputs)

    print('┌────────────┐')
    print('│ embeddings │')
    print('└────────────┘')
    print(f'shape: {tuple(embeddings.shape)}')
