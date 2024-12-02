import torch
from torch import nn
import sys
sys.path.append("../")
from src.board_final import TensorBoardUtilV4
import chess
import os


class BoardAutoencoder(nn.Module):
    LAYER_A = 2048
    LAYER_B = 1024
    LAYER_C = 512
    TARGET_SIZE = 256

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(TensorBoardUtilV4.SIZE, BoardAutoencoder.LAYER_A),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_A, BoardAutoencoder.LAYER_B),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_B, BoardAutoencoder.LAYER_C),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_C, BoardAutoencoder.TARGET_SIZE),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BoardAutoencoder.TARGET_SIZE, BoardAutoencoder.LAYER_C),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_C, BoardAutoencoder.LAYER_B),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_B, BoardAutoencoder.LAYER_A),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_A, TensorBoardUtilV4.SIZE),
        )
    

    def full_decode(self, encoded):
        y = self.decoder(encoded)
        output = TensorBoardUtilV4.discretizeTensor(y)
        return output

    def forward(self, x):
        encoded = self.encoder(x)
        y = self.decoder(encoded)
        return y

    def encode(self, tensor : torch.Tensor) -> torch.Tensor:
        return self.encoder(tensor)
    
    def decode(self, tensor : torch.Tensor) -> torch.Tensor:
        return self.decoder(tensor)

    def encodeFromBoard(self, board : chess.Board) -> torch.Tensor:
        return self.encoder(TensorBoardUtilV4.fromBoard(board))

    def decodeToTensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.full_decode(tensor)

    def decodeToBoard(self, tensor: torch.Tensor) -> chess.Board:
        return TensorBoardUtilV4.toBoard(self.full_decode(tensor))
    


"""
class SmallerAutoencoder(BoardAutoencoder):
    LAYER_A = 2048
    LAYER_B = 1024
    LAYER_C = 512
    TARGET_SIZE = 256

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(TensorBoardUtilV4.SIZE, BoardAutoencoder.LAYER_A),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_A, BoardAutoencoder.LAYER_B),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_B, BoardAutoencoder.LAYER_C),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_C, BoardAutoencoder.TARGET_SIZE),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BoardAutoencoder.TARGET_SIZE, BoardAutoencoder.LAYER_C),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_C, BoardAutoencoder.LAYER_B),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_B, BoardAutoencoder.LAYER_A),
            nn.ReLU(),
            nn.Linear(BoardAutoencoder.LAYER_A, TensorBoardUtilV4.SIZE),
        )
"""

class ProductionAutoencoder(BoardAutoencoder):

    MODEL_DIR = "autoencoder.pth"

    def __init__(self):
        super().__init__()
        assert(ProductionAutoencoder.MODEL_DIR in os.listdir("models"))
        self.load_state_dict(torch.load(f"models/{ProductionAutoencoder.MODEL_DIR}", weights_only=True))



class Evaluator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(ProductionAutoencoder.TARGET_SIZE, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
        )


    def forward(self, boards_tensor: torch.Tensor):
        return self.model(boards_tensor)
    

class ProductionEvaluator(Evaluator):

    MODEL_DIR = "deep-evaluator-mrl"

    def __init__(self):
        super().__init__()
        assert(ProductionEvaluator.MODEL_DIR in os.listdir("models"))
        self.load_state_dict(torch.load(f"models/{ProductionEvaluator.MODEL_DIR}", weights_only=True))


    def forward(self, boards_tensor: torch.Tensor):
        return self.model(boards_tensor)
