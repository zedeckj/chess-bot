import dis
import random
from board_final import TensorBoardUtilV4
from autoencoder import BoardAutoencoder, SMALL_TENSORS
import unittest
import chess
import torch
from dataloader import TENSORS3, load_train_test



def run_and_show(model : BoardAutoencoder, tensor: torch.Tensor):
    board = TensorBoardUtilV4.toBoard(tensor)
    if board.ep_square == None:
        return
    enc = model.encoder(tensor)
    decoded = model.decoder(enc)
    castling_tensor1 = TensorBoardUtilV4.tensorToCastlingRights(tensor)
    castling_tensor2 = torch.sigmoid(TensorBoardUtilV4.tensorToCastlingRights(decoded))
    discrete = TensorBoardUtilV4.discretizeTensor(decoded)
    board2 = TensorBoardUtilV4.toBoard(discrete)
    print(f"\nREAL FEN: {board.fen(en_passant = 'fen')}\nPREDICTED FEN: {board2.fen(en_passant = 'fen')}\n{board}\n\n\n{board2}")
    print(f"\nREAL CASTLING: {castling_tensor1}\nPREDICTED CASTLING: {castling_tensor2}")



def main():
    model = BoardAutoencoder()
    model.load_state_dict(torch.load("models/128-autoencoder.pth", weights_only=True))
    _, test_dataset = load_train_test(TENSORS3, 1, 0.1)
    for t in test_dataset:
        run_and_show(model, t)


    
main()

def test_loss():
    ...
    
