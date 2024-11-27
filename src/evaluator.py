import torch
from torch import nn
from autoencoder import BoardAutoencoder
class Evaluator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(BoardAutoencoder.TARGET_SIZE, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
        )


    def forward(self, boards_tensor: torch.Tensor):
        return self.model(boards_tensor)
    

class EvaluatorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()
        self.softmin = nn.Softmin()
        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, evaluations : torch.Tensor):
        if for_white:
            evaluations = self.softmax(evaluations)
        else:
            evaluations = self.softmin(evaluations)
        return self.ce_loss(evaluations, best_move)
    




        
    