import numpy
import torch

class EarlyStopping:
    def __init__(self, epochs, patience=7, delta=0, output_dir='checkpoint', best_score=numpy.inf):
        self.epochs = epochs
        self.patience = patience
        self.delta = delta
        self.path = output_dir
        self.counter = 0
        self.early_stop = False
        self.best_score = best_score
    def condition(self, score):
        return False
    def update(self, model, score):
        if self.condition(score):
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), f'{self.path}/best.pt')
        else:
            self.counter += 1
            print(f'EarlyStopping {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
    def __iter__(self):
        for epoch in range(1, self.epochs + 1):
            yield epoch
            if self.early_stop:
                print(f'Early stopped at epoch {epoch}')
                break

class MaximizeEarlyStopping(EarlyStopping):
    def __init__(self, epochs, patience=7, delta=0, output_dir='checkpoint'):
        super().__init__(epochs, patience, delta, output_dir, -numpy.inf)
    def condition(self, score):
        return score > self.best_score + self.delta

class MinimizeEarlyStopping(EarlyStopping):
    def __init__(self, epochs, patience=7, delta=0, output_dir='checkpoint'):
        super().__init__(epochs, patience, delta, output_dir, numpy.inf)
    def condition(self, score):
        return score < self.best_score - self.delta