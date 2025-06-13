import seaborn
import torch
import warnings

from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

def evaluate(model, loader):
    model.eval()
    ys = []
    predicts = []
    for x, y in tqdm(loader):
        with torch.no_grad():
            logits = model(x)

        ys.append(y)
        predicts.append(logits)

    ys = torch.cat(ys, dim=0)
    predicts = torch.cat(predicts, dim=0)
    return ys, predicts

def plot_confusion_matrix(ys, predicts, labels, output_folder='checkpoint'):
    confusion = confusion_matrix(ys, predicts)

    pyplot.figure(figsize=(8, 6))
    seaborn.heatmap(
        confusion,
        cmap='Blues',
        annot=True,
        fmt='d',
        square=True,
        xticklabels=labels,
        yticklabels=labels
    )
    pyplot.yticks(rotation=0)
    pyplot.xlabel('Predict')
    pyplot.ylabel('True')
    pyplot.title('Confusion Matrix')
    pyplot.tight_layout()
    pyplot.savefig(f'{output_folder}/confusion.png')

def map_k(ys, predicts, k=3):
    labels = predicts.topk(k, dim=-1).indices
    map_k = 0
    for y, labels in zip(ys, labels):
        for index, label in enumerate(labels):
            if y == label:
                map_k += 1 / (index + 1)
                break
    map_k /= len(ys)
    return map_k

if __name__ == '__main__':
    from dataset import get_loader
    from load_data import load_data
    from model import Classifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    torch.manual_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../playground-series-s5e6/train.csv')

    model = Classifier(categorical_features, numerical_features, num_class=7).cuda()
    state_dict = torch.load('checkpoint/best.pt', weights_only=True)
    model.load_state_dict(state_dict)

    loader = get_loader(valid_dataset, batch_size=256, shuffle=False)

    ys, predicts = evaluate(model, loader)

    map_3 = map_k(ys, predicts)

    predicts = predicts.cpu().argmax(-1)
    accuracy = accuracy_score(ys, predicts)
    precision = precision_score(ys, predicts, average='macro', zero_division=0)
    recall = recall_score(ys, predicts, average='macro')
    f1 = f1_score(ys, predicts, average='macro')
    print('┌───────────┬────────┐')
    print(f'│ MAP@3     │ {map_3*100:5.2f}% │')
    print('├───────────┼────────┤')
    print(f'│ Accuracy  │ {accuracy*100:5.2f}% │')
    print(f'│ Precision │ {precision*100:5.2f}% │')
    print(f'│ Recall    │ {recall*100:5.2f}% │')
    print(f'│ F1 Score  │ {f1*100:5.2f}% │')
    print('└───────────┴────────┘')

    labels = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
    plot_confusion_matrix(ys, predicts, labels)
