import torch
import pandas as pd
from model_CLIP import CLIPClassifier  
from feature_extractor_CLIP import CLIPFeatureExtractor
from feature_processor_CLIP import CLIPFeatureProcessor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

model = CLIPClassifier(
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    num_class=num_class
).to(device)

state_dict = torch.load("checkpoint/best.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

feature_extractor = CLIPFeatureExtractor(
    categorical_features=categorical_features,
    numerical_features=numerical_features,
)

fertilizer_names = []
batch_size = 128

with torch.no_grad():
    for i in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[i:i+batch_size]
        x = batch_df[feature_columns]
        
        x_emb = feature_extractor(x, device=device)  
        x_emb = model.base_model.feature_processor(x_emb)  
        x_emb = model.base_model.cls_token(x_emb)         
        x_encoded = model.base_model.encoder(x_emb)

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
submission_df.to_csv("submission_clip.csv", index=False)

print("submission.csv 已完成")
