import torch
from torch import nn
from board_final import TensorBoardUtilV4
import chess
import os


class BoardAutoencoder(nn.Module):
    LAYER_A = 2048
    LAYER_B = 1024
    LAYER_C = 512
    TARGET_SIZE = 128

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
    

    def full_decode(self, encoded):
        y = self.decoder(encoded)
        output = TensorBoardUtilV4.discretizeTensor(y)
        return output

    def forward(self, x):
        encoded = self.encoder(x)
        y = self.decoder(encoded)
        return y
    
    def full_output(self, x):
        encoded = self.encoder(x)
        return self.full_decode(encoded)

    def encodeFromTensor(self, tensor : torch.Tensor) -> torch.Tensor:
        return self.encoder(tensor)
    
    def encodeFromBoard(self, board : chess.Board) -> torch.Tensor:
        return self.encoder(TensorBoardUtilV4.fromBoard(board))

    def decodeToTensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.full_decode(tensor)

    def decodeToBoard(self, tensor: torch.Tensor) -> chess.Board:
        return TensorBoardUtilV4.toBoard(self.full_decode(tensor))
    

class ProductionAutoencoder(BoardAutoencoder):

    MODEL_DIR = "complete-128-autoencoder.pth"

    def __init__(self):
        super().__init__()
        assert(ProductionAutoencoder.MODEL_DIR in os.listdir("models"))
        self.load_state_dict(torch.load(f"models/{ProductionAutoencoder.MODEL_DIR}", weights_only=True))

