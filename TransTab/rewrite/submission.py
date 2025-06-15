import torch
import pandas as pd
from model import Classifier
from feature_extractor import FeatureExtractor
from feature_processor import FeatureProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label = {
    0: '10-26-26',
    1: '14-35-14',
    2: '17-17-17',
    3: '20-20',
    4: '28-28',
    5: 'DAP',
    6: 'Urea'
}
num_class = len(label)

categorical_features = ['Soil Type', 'Crop Type']
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

test_df = pd.read_csv("../playground-series-s5e6/test.csv")
feature_columns = categorical_features + numerical_features

model = Classifier(
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    num_class=num_class
)
model.load_state_dict(torch.load("checkpoint/best.pt", map_location=device,weights_only=True))
model.to(device)
model.eval()

all_ids = []
fertilizer_names = []

batch_size = 128

with torch.no_grad():
    for i in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[i:i+batch_size]
        x = batch_df[feature_columns]

        x_tokenized = model.base_model.feature_extractor(x)
        x_processed = model.base_model.feature_processor(**x_tokenized)
        x_emb = model.base_model.cls_token(**x_processed)
        x_encoded = model.base_model.encoder(**x_emb)

        logits = model.classifier(x_encoded[:, 0, :])
        probs = torch.softmax(logits, dim=1)
        top3 = torch.topk(probs, k=3, dim=1).indices.cpu().numpy()

        for preds in top3:
            names = [label[p] for p in preds]
            fertilizer_names.append(" ".join(names))

submission_df = pd.DataFrame({
    "id": test_df["id"].values,
    "Fertilizer Name": fertilizer_names
})
submission_df.to_csv("submission.csv", index=False)

print("submission.csv已完成")
