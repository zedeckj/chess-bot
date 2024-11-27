from abc import ABC, abstractmethod
import torch
from torch import nn, relu

from Common.autoencoder import BoardAutoencoder

class AbstractTargetEngine(ABC):
    """
    Represents any engine that can provide a UCI best move based on a given FEN  
    """

    @abstractmethod
    def best_move(self, uci_move : str):
        ...

class MicroEvaluator(nn.Module):
    """
    This class is a proof of concept of using a deeper evaluation function. This model is intended to be trained 
    simply on the evaluations of a strong engine.
    """

    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(BoardAutoencoder.TARGET_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )



class MicroEvaluatorTrainer(nn.Module):

    def __init__(self):
        ...



