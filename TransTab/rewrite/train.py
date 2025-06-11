import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, output_dir='ckpt', trace_func=print, less_is_better=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print     
            less_is_better (bool): If True (e.g., val loss), the metric is less the better.       
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = output_dir
        self.trace_func = trace_func
        self.less_is_better = less_is_better

    def __call__(self, val_loss, model):
        if self.patience < 0: # no early stop
            self.early_stop = False
            return
        
        if self.less_is_better:
            score = val_loss
        else:    
            score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, 'model.pt'))
        self.val_loss_min = val_loss

class TrainDataset(Dataset):
    def __init__(self, dataset):
        self.x, self.y = dataset
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitems__(self, indices):
        x = self.x.iloc[indices]
        y = self.y.iloc[indices]
        return x, y

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

class Trainer:
    def __init__(self,
        model,
        train_set_list,
        test_set_list=None,
        output_dir='./ckpt',
        num_epoch=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=0,
        patience=5,
        eval_batch_size=256,
        warmup_ratio=None,
        warmup_steps=None,
        balance_sample=False,
        load_best_at_last=True,
        eval_metric='auc',
        eval_less_is_better=False,
        num_workers=0,
        ):
        '''args:
        train_set_list: a list of training sets [(x_1,y_1),(x_2,y_2),...]
        test_set_list: a list of tuples of test set (x, y), same as train_set_list. if set None, do not do evaluation and early stopping
        patience: the max number of early stop patience
        num_workers: how many workers used to process dataloader. recommend to be 0 if training data smaller than 10000.
        eval_less_is_better: if the set eval_metric is the less the better. For val_loss, it should be set True.
        '''
        self.model = model
        if isinstance(train_set_list, tuple):
            train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple):
            test_set_list = [test_set_list]

        self.train_set_list = train_set_list
        self.test_set_list = test_set_list
        self.collate_fn = lambda x: x
        self.trainloader_list = [
            self._build_dataloader(trainset, batch_size, collator=self.collate_fn, num_workers=num_workers) for trainset in train_set_list
        ]
        if test_set_list is not None:
            self.testloader_list = [
                self._build_dataloader(testset, eval_batch_size, collator=self.collate_fn, num_workers=num_workers, shuffle=False) for testset in test_set_list
            ]
        else:
            self.testloader_list = None
            
        self.test_set_list = test_set_list
        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(output_dir=output_dir, patience=patience, verbose=False, less_is_better=eval_less_is_better)
        self.args = {
            'lr':lr,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'num_epoch':num_epoch,
            'eval_batch_size':eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'num_training_steps': self.get_num_train_steps(train_set_list, num_epoch, batch_size),
            'eval_metric': lambda y, p: accuracy_score(y, np.argmax(p, -1)),
            'eval_metric_name': eval_metric,
            }
        self.args['steps_per_epoch'] = int(self.args['num_training_steps'] / (num_epoch*len(self.train_set_list)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.optimizer = None
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

    def train(self):
        args = self.args
        self.create_optimizer()
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(1, args['num_epoch'] + 1):
            ite = 0
            train_loss_all = 0
            for dataindex in range(len(self.trainloader_list)):
                for data in tqdm(self.trainloader_list[dataindex]):
                    self.optimizer.zero_grad()
                    logits = self.model(data[0])
                    y = torch.tensor(data[1].values).long()
                    loss = loss_fn(logits, y)
                    loss.backward()
                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            if self.test_set_list is not None:
                eval_res_list = self.evaluate()
                eval_res = np.mean(eval_res_list)
                print('epoch: {}, test {}: {:.6f}'.format(epoch, self.args['eval_metric_name'], eval_res))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print('early stopped')
                    break
            print('epoch: {}, train loss: {:.4f}, lr: {:.6f}'.format(epoch, train_loss_all, self.optimizer.param_groups[0]['lr']))

        if os.path.exists(self.output_dir):
            if self.test_set_list is not None:
                # load checkpoints
                state_dict = torch.load(os.path.join(self.output_dir, 'model.pt'), map_location='cpu', weights_only=True)
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

    def evaluate(self):
        # evaluate in each epoch
        self.model.eval()
        eval_res_list = []
        loss_fn = nn.CrossEntropyLoss()
        for dataindex in range(len(self.testloader_list)):
            y_test, pred_list, loss_list = [], [], []
            for data in self.testloader_list[dataindex]:
                if data[1] is not None:
                    label = data[1]
                    if isinstance(label, pd.Series):
                        label = label.values
                    y_test.append(label)
                with torch.no_grad():
                    logits = self.model(data[0])
                    y = torch.tensor(data[1].values).long()
                    loss = loss_fn(logits, y)
                if loss is not None:
                    loss_list.append(loss.item())
                if logits is not None:
                    if logits.shape[-1] == 1: # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else: # multi-class classification
                        pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())

            if len(pred_list)>0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()

            if self.args['eval_metric_name'] == 'val_loss':
                eval_res = np.mean(loss_list)
            else:
                y_test = np.concatenate(y_test, 0)
                eval_res = self.args['eval_metric'](y_test, pred_all)

            eval_res_list.append(eval_res)

        return eval_res_list

    def save_model(self, output_dir=None):
        if output_dir is None:
            print('no path assigned for save mode, default saved to ./ckpt/model.pt !')
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, f'{output_dir}/model.pt')

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
        if self.args is not None:
            train_args = {}
            for k,v in self.args.items():
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float):
                    train_args[k] = v
            with open(os.path.join(output_dir, 'training_args.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(train_args))

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args['lr'])

    def get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        total_step *= num_epoch
        return total_step

    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=True):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            )
        return trainloader

if __name__ == '__main__':
    import sys
    sys.path.append('..')

    import transtab
    from model import Classifier

    transtab.random_seed(42)

    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, binary_features = transtab.load_data('../dataset')
    print('########################################')

    model = Classifier(categorical_features, numerical_features, binary_features, num_class=7)
    print('########################################')

    x, y = valid_dataset

    xx = x.head(500)
    xy = y.head(500)
    yx = x.tail(500)
    yy = y.tail(500)

    train_dataset = (xx, xy)
    valid_dataset = (yx, yy)

    train_dataset = [train_dataset]

    trainer = Trainer(
        model,
        train_dataset,
        valid_dataset,
        output_dir='checkpoint',
        num_epoch=2,
        batch_size=256,
        lr=1e-4,
        weight_decay=0,
        patience=5,
        eval_batch_size=256,
        warmup_ratio=None,
        warmup_steps=None,
        balance_sample=False,
        load_best_at_last=True,
        eval_metric='acc',
        eval_less_is_better=False,
        num_workers=0
    )
    trainer.train()
