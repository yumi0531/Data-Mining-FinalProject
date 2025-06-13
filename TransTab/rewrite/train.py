import torch
import warnings

from sklearn.metrics import accuracy_score
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

def train_epoch(model, loader, epoch, loss_fn, optimizer):
    model.train()

    losses = 0
    accuracies = 0

    for x, y in tqdm(loader, desc=f'Epoch {epoch}'):
        batch_size = len(x)

        logits = model(x)
        loss = loss_fn(logits, y.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item() * batch_size
        predicts = logits.detach().cpu().argmax(-1)
        accuracy = accuracy_score(y, predicts)
        accuracies += accuracy * batch_size
    
    avg_loss = losses / len(loader.dataset)
    avg_accuracy = accuracies / len(loader.dataset)
    return avg_loss, avg_accuracy

def valid_epoch(model, loader, epoch, loss_fn):
    model.eval()

    losses = 0
    accuracies = 0

    space = ' ' * len(f'Epoch {epoch}')
    for x, y in tqdm(loader, desc=space):
        batch_size = len(x)

        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y.cuda())

        losses += loss.item() * batch_size
        predicts = logits.cpu().argmax(-1)
        accuracy = accuracy_score(y, predicts)
        accuracies += accuracy * batch_size

    avg_loss = losses / len(loader.dataset)
    avg_accuracy = accuracies / len(loader.dataset)
    return avg_loss, avg_accuracy

if __name__ == '__main__':
    import os

    from dataset import get_loader
    from early_stopping import MinimizeEarlyStopping
    from load_data import load_data
    from matplotlib import pyplot
    from model import Classifier
    from optimizer import get_optimizer
    from torch import nn

    torch.manual_seed(42)

    train_batch_size = 256
    valid_batch_size = 256
    output_folder = 'checkpoint'
    epochs = 10
    learning_rate = 1e-4
    weight_decay=0

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../playground-series-s5e6/train.csv')

    model = Classifier(categorical_features, numerical_features, num_class=7).cuda()

    # use only the first and last 500 rows for fast testing
    # x, y = valid_dataset
    # train_x = x.head(500)
    # train_y = y.head(500)
    # valid_x = x.tail(500)
    # valid_y = y.tail(500)

    # train_dataset = (train_x, train_y)
    # valid_dataset = (valid_x, valid_y)

    train_loader = get_loader(train_dataset, batch_size=256)
    valid_loader = get_loader(valid_dataset, batch_size=256, shuffle=False)

    os.makedirs(output_folder, exist_ok=True)

    early_stopping = MinimizeEarlyStopping(epochs, patience=5, output_dir=output_folder)

    optimizer = get_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss().cuda()

    train_losses = []
    valid_losses = []

    # for epoch in range(1, epochs + 1):
    for epoch in early_stopping:
        train_loss, train_accuracy = train_epoch(model, train_loader, epoch, loss_fn, optimizer)
        valid_loss, valid_accuracy = valid_epoch(model, valid_loader, epoch, loss_fn)

        space = ' ' * len(f'Epoch {epoch}')
        print(f'{space} train loss: {train_loss:.6f}, train accuracy: {train_accuracy*100:.2f}%')
        print(f'{space} valid loss: {valid_loss:.6f}, valid accuracy: {valid_accuracy*100:.2f}%')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        early_stopping.update(model, valid_loss)

    torch.save(model.state_dict(), f'{output_folder}/final.pt')
    torch.save(optimizer.state_dict(), f'{output_folder}/optimizer.pt')

    epochs = range(1, len(train_losses) + 1)
    pyplot.plot(epochs, train_losses, label='train')
    pyplot.plot(epochs, valid_losses, label='valid')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig(f'{output_folder}/loss.png')
