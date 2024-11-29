import dis
import random
import sys
sys.path.append("./")
from src.board_final import BoardGenerator, TensorBoardUtilV4
from src.chess_models import ProductionAutoencoder
import unittest
import chess
import torch
from dataloader import TENSORS3, load_train_test



def run_and_show(model : ProductionAutoencoder, board : chess.Board):
    tensor = TensorBoardUtilV4.fromBoard(board)
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
    model = ProductionAutoencoder()
    for board in BoardGenerator(10000):
        run_and_show(model, board)


main()

def test_loss():
    ...
    
