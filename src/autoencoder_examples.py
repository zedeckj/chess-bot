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
from autoencoder import BoardLoss
import pickle

class TestBoardLoss(unittest.TestCase):

    def setUp(self):
        data_filename = "board_tensorsV3/2016-03/0_0.tnsrs"
        with open(data_filename, "rb") as f:
            data = pickle.load(f)
        self.loss_fn = BoardLoss(data)
        self.data = data

    def testPieceCountLoss(self):
        board = chess.Board()
        tensor = TensorBoardUtilV4.fromBoard(board)
        self.data

def run_and_show(model : ProductionAutoencoder, board : chess.Board):
    VERBOSE = False
    tensor = TensorBoardUtilV4.fromBoard(board)
    if board.ep_square == None:
        return
    enc = model.encoder(tensor)
    decoded = model.decoder(enc)
    castling_tensor1 = TensorBoardUtilV4.tensorToCastlingRights(tensor)
    castling_tensor2 = torch.sigmoid(TensorBoardUtilV4.tensorToCastlingRights(decoded))
    pieces_decoded = TensorBoardUtilV4.tensorToPieceTensors(decoded)
    pieces_probs = torch.softmax(pieces_decoded, dim = -1)
    print(pieces_probs.shape)
    discrete = TensorBoardUtilV4.discretizeTensor(decoded)
    board2 = TensorBoardUtilV4.toBoard(discrete)
    print(f"\nREAL FEN: {board.fen(en_passant = 'fen')}\nPREDICTED FEN: {board2.fen(en_passant = 'fen')}\n{board}\n\n\n{board2}")
    print(f"\nREAL CASTLING: {castling_tensor1.tolist()}\nPREDICTED CASTLING: {castling_tensor2.tolist()}")
    if VERBOSE:
        for i in range(64):
            print(f"Square {i} has a {board.piece_at(i)}")
            print([f"{p * 100:.2f}%" for p in pieces_probs[0,i,:].tolist()])

def main():
    model = ProductionAutoencoder()
    for board in BoardGenerator(10000):
        run_and_show(model, board)


main()

def test_loss():
    ...
    
