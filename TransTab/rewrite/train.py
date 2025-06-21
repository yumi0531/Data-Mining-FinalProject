import torch
import warnings

from sklearn.metrics import accuracy_score
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from captum.attr import IntegratedGradients
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)


def train_epoch(model, loader, epoch, loss_fn, optimizer):
    model.train()

    losses = 0
    accuracies = 0

    for x, y in tqdm(loader, desc=f'Epoch {epoch}'):
        batch_size = len(x)

        logits = model(x)
        loss = loss_fn(logits, y.cuda())
        # logits, loss = model(x, y)
        
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
            # logits, loss = model(x, y)
            
        losses += loss.item() * batch_size
        predicts = logits.cpu().argmax(-1)
        accuracy = accuracy_score(y, predicts)
        accuracies += accuracy * batch_size

    avg_loss = losses / len(loader.dataset)
    avg_accuracy = accuracies / len(loader.dataset)
    return avg_loss, avg_accuracy



def explain_feature_importance(model, x_df, output_folder, log_path=None, target_class=None, sample_size=32):
    model.eval()
    device = next(model.parameters()).device

    # 選取一小部分資料
    x_sample = x_df.head(sample_size)

    # 透過 base_model 處理特徵（CLIP embedding → linear projection）
    embedding = model.base_model.feature_extractor(x_sample, device)
    x_tensor = model.base_model.feature_processor(embedding)

    # 加上 [CLS] token
    x_tensor = model.base_model.cls_token(x_tensor)

    # 若未指定 target_class，則預測後自動選取第一個 sample 的最大值
    if target_class is None:
        with torch.no_grad():
            hidden = model.base_model.encoder(x_tensor)
            logits = model.classifier(hidden[:, 0])
            target_class = torch.argmax(logits, dim=1)[0].item()

    # 包裝 forward 函數供 Captum 使用
    def model_forward(input_tensor):
        hidden = model.base_model.encoder(input_tensor)
        logits = model.classifier(hidden[:, 0])
        probs = torch.softmax(logits, dim=-1)
        return probs[:, target_class]

    # 計算 attribution
    ig = IntegratedGradients(model_forward)
    attributions, _ = ig.attribute(inputs=x_tensor, return_convergence_delta=True)

    # 每個 token 維度平均
    importances = attributions.abs().mean(dim=0).mean(dim=1).detach().cpu().numpy()
    feature_names = [f'Feature_{i}' for i in range(x_tensor.shape[1])]
    df_attr = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # 顯示前10名
    print(f"\nCaptum Feature Importance (class {target_class}):")
    print(df_attr.head(10).round(4))

    # 繪圖
    plt.figure(figsize=(10, 6))
    df_attr.head(15).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(f"Integrated Gradients (Class {target_class})")
    plt.xlabel("Attribution Magnitude")
    plt.tight_layout()
    plt.savefig(f'{output_folder}/captum_feature_importance_class{target_class}.png')
    print(f"PNG saved to: {output_folder}/captum_feature_importance_class{target_class}.png")

    # 寫入 summary.txt
    if log_path:
        with open(log_path, 'a') as f:
            f.write("\n")
            f.write("=" * 10 + f" Captum Feature Importance (class {target_class}) " + "=" * 10 + "\n")
            f.write(df_attr.head(10).round(4).to_string())
            f.write("\n")

if __name__ == '__main__':
    import os

    from dataset import get_loader
    from early_stopping import MinimizeEarlyStopping
    from load_data_mush import load_data
    from matplotlib import pyplot
    from model_CLIP import CLIPClassifier
    from optimizer import get_optimizer
    from torch import nn
    from model import Classifier
    from omegaconf import OmegaConf
    from datetime import datetime

    torch.manual_seed(42)
    
    """ Training Configuration """
    cfg = OmegaConf.load("training_config.yml")
    train_batch_size = cfg.train.batch_size
    valid_batch_size = cfg.train.batch_size
    
    output_folder = cfg.output_folder
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = os.path.join(cfg.output_folder, time_stamp)
    
    epochs = cfg.train.epochs
    patience = cfg.train.patience
    dropout = cfg.train.dropout
    learning_rate = cfg.train.learning_rate
    weight_decay = cfg.train.weight_decay
    use_CLIP = cfg.train.use_CLIP
    use_nan_embedding = cfg.train.use_NaN_embedding
    fast_testing = cfg.train.fast_training_for_test
    num_class = 7 if cfg.train.dataset == "fertilizer" else 2
    """"""

    if cfg.train.dataset == "fertilizer":
        load_data_path = '../playground-series-s5e6/train.csv'
    else:
        load_data_path = '../MushroomDataset/secondary_data.csv'

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data(
        load_data_path)

    if use_CLIP:
        model = CLIPClassifier(categorical_features, numerical_features, num_class=num_class, hidden_dropout_prob=dropout).cuda()
    else:
        model = Classifier(categorical_features,numerical_features,num_class=num_class,hidden_dropout_prob=dropout, use_nan_embedding=use_nan_embedding).cuda()


    # use only the first and last 500 rows for fast testing
    if fast_testing:
        x, y = valid_dataset
        train_x = x.head(500)
        train_y = y.head(500)
        valid_x = x.tail(500)
        valid_y = y.tail(500)

        train_dataset = (train_x, train_y)
        valid_dataset = (valid_x, valid_y)

    train_loader = get_loader(train_dataset, batch_size=train_batch_size)
    valid_loader = get_loader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    os.makedirs(output_folder, exist_ok=True)
    
    log_path = os.path.join(output_folder, "summary.txt")

    # === record timestamp & config ===
    with open(log_path, 'w') as f:
        f.write(f"Training Summary - {time_stamp}\n")
        f.write("\n")
        f.write("=" * 10 + "Training Parameter" + "=" * 10 + "\n")
        f.write(OmegaConf.to_yaml(cfg))

    # === Train Loop ===
    early_stopping = MinimizeEarlyStopping(epochs, patience=patience, output_dir=output_folder)

    optimizer = get_optimizer(model,
                              learning_rate=learning_rate,
                              weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss().cuda()

    train_losses = []
    valid_losses = []
    best_record = None

    # for epoch in range(1, epochs + 1):
    for epoch in early_stopping:
        train_loss, train_accuracy = train_epoch(model, train_loader, epoch, loss_fn, optimizer)
        valid_loss, valid_accuracy = valid_epoch(model, valid_loader, epoch, loss_fn)

        space = ' ' * len(f'Epoch {epoch}')
        print(f'{space} train loss: {train_loss:.6f}, train accuracy: {train_accuracy*100:.2f}%')
        print(f'{space} valid loss: {valid_loss:.6f}, valid accuracy: {valid_accuracy*100:.2f}%')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        early_stopping.update(model, valid_loss, epoch=epoch)
        
        # record the best model info to txt file
        if early_stopping.best_epoch == epoch:
            best_record = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accuracy,
                'val_loss': valid_loss,
                'val_acc': valid_accuracy,
            }

    # === Save Model ===
    torch.save(model.state_dict(), f'{output_folder}/final.pt')
    torch.save(optimizer.state_dict(), f'{output_folder}/optimizer.pt')

    # === Save loss plot ===
    epochs = range(1, len(train_losses) + 1)
    pyplot.plot(epochs, train_losses, label='train')
    pyplot.plot(epochs, valid_losses, label='valid')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig(f'{output_folder}/loss.png')
    
    # === Save feature importance ===
    # valid_x, _ = valid_dataset
    # explain_feature_importance(model, valid_x, output_folder, log_path=log_path)
    
    # === Save best model info to txt file ===
    if best_record:
        with open(log_path, 'a') as f:
            f.write("\n")
            f.write("=" * 10 + "Best Epoch Summary" + "=" * 10 + "\n")
            f.write(f"Best Epoch: {best_record['epoch']:03d}\n")
            
            f.write(f"Train Loss: {best_record['train_loss']:.4f}, "
                    f"Train Acc: {best_record['train_acc']:.2f}%\n")
            
            f.write(f"Val   Loss: {best_record['val_loss']:.4f}, "
                    f"Val   Acc: {best_record['val_acc']:.2f}%\n")
