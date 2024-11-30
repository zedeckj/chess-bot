from ctypes.wintypes import LARGE_INTEGER
from chess import Board
import torch
from torch import Tensor, nn
import os
import sys

from src.trainer import Trainer
sys.path.append("./")
import tqdm
from src.chess_models import BoardAutoencoder
from src.board_final import TensorBoardUtilV4
from src.dataloader import DATASET_2016_03, DIR_2016_03, LARGER_TENSORS, LARGER_TENSORS2, SMALL_TENSORS, TENSORS3, DIR_2013_12
from src.dataloader import load_tensor as dataloader_load_tensor
from src.eval_tools import display_precision_recall
from src.torch_utils import unbatch
import random
import chess

import unittest
from board_final import BoardGenerator
import itertools
from typing import Callable, Optional


MODEL_NAME = "128-autoencoder.pth"

DEVICE = "mps"

class BoardLoss(nn.Module):


    def _generate_weights(self, dataset : torch.Tensor):
        with torch.no_grad():


            positives = torch.zeros((1,dataset.shape[2]), device = DEVICE)
            bitmask = torch.ones(TensorBoardUtilV4.BINARY_RANGE, device = DEVICE)
            mask = torch.unsqueeze(torch.cat([bitmask, torch.zeros(2, device = DEVICE)]),0)
            for i in tqdm.tqdm(range(len(dataset)), desc = "Generating Loss Weights"):
                tensor = dataset[i]
                tensor = tensor.to(DEVICE)
                tensor = mask * tensor
                positives += torch.sum(tensor, dim = 0) 
            lengths = dataset.shape[0] * dataset.shape[1]

            bce_out = (lengths - positives)/(positives + 1e-8)
            ce_out = 1/(positives + 1e-14)
            pieces_weight = TensorBoardUtilV4.tensorToPieceTensors(ce_out)
            turn_weight = torch.squeeze(TensorBoardUtilV4.tensorToTurn(bce_out), dim = 0)
            castling_weight = torch.squeeze(TensorBoardUtilV4.tensorToCastlingRights(bce_out), dim = 0)
            castling_weight = torch.squeeze(TensorBoardUtilV4.tensorToCastlingRights(bce_out), dim = 0)
            castling_positives = TensorBoardUtilV4.tensorToCastlingRights(positives)
            en_passant_weight = torch.squeeze(TensorBoardUtilV4.tensorToEnPassant(ce_out), dim = 0)
            pieces_loss = []
            for weight in pieces_weight[0]:
                pieces_loss.append(nn.CrossEntropyLoss(weight = weight))

            # Pieces have been learned very well, but castling and turns seem not to.
            turn_loss = nn.BCEWithLogitsLoss(pos_weight = turn_weight)
            castling_loss = nn.BCEWithLogitsLoss(pos_weight = castling_weight)
            en_passant_loss = nn.CrossEntropyLoss(weight = en_passant_weight)
            clock_loss = nn.L1Loss()

            return pieces_loss, turn_loss, castling_loss, en_passant_loss, clock_loss        


    def __init__(self, train_dataset):
        super(BoardLoss, self).__init__()
        pieces_loss, turn_loss, castling_loss, en_passant_loss, clock_loss = self._generate_weights(train_dataset)
        self.pieces_loss = pieces_loss
        self.turn_loss = turn_loss
        self.castling_loss = castling_loss
        self.en_passant_loss = en_passant_loss
        self.clock_loss = clock_loss



        self.pieces_count_targets = self._create_piece_loss_targets()



    def _create_piece_loss_targets(self):
        targets = torch.zeros(13)
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.WHITE))] = 16
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.BLACK))] = 16

        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KNIGHT, chess.WHITE))] = 2
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KNIGHT, chess.BLACK))] = 2

        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.BISHOP, chess.WHITE))] = 2
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.BISHOP, chess.BLACK))] = 2

        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KING, chess.WHITE))] = 1
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KING, chess.BLACK))] = 1

        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.ROOK, chess.WHITE))] = 2
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.ROOK, chess.BLACK))] = 2

        targets[TensorBoardUtilV4.indexOfPiece(None)] = 32
        targets[TensorBoardUtilV4.indexOfPiece(None)] = 32

        # Queens remain at zero, as they will be excluded from the loss utilizing this tensor

        return targets

        





    def piece_count_loss_fn(self, output : torch.Tensor) -> torch.Tensor:
        """
        Since there is never more than 16 pawns, and never more than 1 king for each color,
        we can add a loss term for predicting there are more than this number. We use the sum of probabilties
        to calculate the projected count. 

        We also include bishops, rooks, and knights as having no more than 2. 
        Altough, this is innacuarate in edge cases due to the existence of underpromotion. However,
        the usage of under promotion while also having two other instances of the target piece is 
        too rare of an occurance to bother with, and likely never appears in the training set.

        """
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        # unsure about these dimensions
        pieces_softmax = torch.softmax(pieces_output, dim = 1) 
        pieces_summed = torch.sum(pieces_softmax, dim = 1)
        # we expect a shape of [N, 13] at this point
        pieces_summed[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.QUEEN, chess.WHITE))] = 0
        pieces_summed[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.QUEEN, chess.BLACK))] = 0

        differences = (pieces_summed - self.pieces_count_targets) ** 3
        zeroes = torch.zeros(differences.shape)
        return torch.maximum(differences, zeroes)

        """
        # THIS CAN BE PARALLIZED MUCH BETTER
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        # unsure about these dimensions
        pieces_softmax = torch.softmax(pieces_output, dim = 1) 
        pieces_summed = torch.sum(pieces_softmax, dim = 1)
        # we expect a shape of [N, 13] at this point

        # drop queens from our calculation
        pieces_summed[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.QUEEN, chess.WHITE))] = 0
        pieces_summed[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.QUEEN, chess.BLACK))] = 0

        white_king_count_loss = 0
        black_king_count_loss = 0
        king_count_loss = white_king_count_loss + black_king_count_loss

        white_pawn_count_loss = 0
        black_pawn_count_loss = 0
        pawn_count_loss = white_pawn_count_loss + black_pawn_count_loss

        white_knight_count_loss = 0
        black_knight_count_loss = 0
        knight_count_loss = white_knight_count_loss + black_knight_count_loss

        white_bishop_count_loss = 0
        black_bishop_count_loss = 0
        bishop_count_loss = white_bishop_count_loss + black_bishop_count_loss

        white_rook_count_loss = 0
        black_rook_count_loss = 0
        rook_count_loss = white_bishop_count_loss + black_bishop_count_loss
        
        # Queens are noticeable excluded, as pawn promotion to a queen is not an edge case at all

        return (
            king_count_loss +
            pawn_count_loss +
            knight_count_loss +
            bishop_count_loss 
        )
        """


            


    def pices_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for piece placement conditions, using 64 distinct CE losses. Revised slightly to parallize generating class labels. 
        Since there exists a unique loss function for each piece-square, this part cannot be parallelized. 
        """
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        pieces_target = TensorBoardUtilV4.tensorToPieceTensors(target)
        class_labels = torch.argmax(pieces_target, dim = 1)
        piece_loss_list = []
        for i in range(64):
            piece_loss_list.append(self.pieces_loss[i](pieces_output[...,i, :], class_labels[i]))
        return torch.stack(piece_loss_list).sum()

    def pices_loss_fn_old(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        pieces_target = TensorBoardUtilV4.tensorToPieceTensors(target)
        piece_loss_list = []
        for i in range(64):
            class_label = torch.argmax(pieces_target[...,i, :], dim = 1)
            piece_loss_list.append(self.pieces_loss[i](pieces_output[...,i, :], class_label))
        return torch.stack(piece_loss_list).sum()
    
    def turn_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for whose turn it is, which is encoded as a single logit, using Binary Cross Entroy.
        """
        turn_output = TensorBoardUtilV4.tensorToTurn(output)
        turn_target = TensorBoardUtilV4.tensorToTurn(target)
        loss = self.turn_loss(turn_output, turn_target)
        return loss

    def castling_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for castling rights, using Binary Cross Entropy for the 4 castling bits.
        """
        castling_output = TensorBoardUtilV4.tensorToCastlingRights(output)
        castling_target = TensorBoardUtilV4.tensorToCastlingRights(target)
        loss = self.castling_loss(castling_output, castling_target)
        return loss
        
    def en_passant_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the en passant square, using Cross Entropy Loss over the 65 logits.
        """
        en_passant_output = TensorBoardUtilV4.tensorToEnPassant(output)
        en_passant_target = torch.argmax(TensorBoardUtilV4.tensorToEnPassant(target), dim = 1)
        return self.en_passant_loss(en_passant_output, en_passant_target)
                                    
    def clock_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the half-move and full-move clocks using L1 loss, or MAE.
        """
        clock_output = TensorBoardUtilV4.tensorToTimers(output)
        clock_target = TensorBoardUtilV4.tensorToTimers(target)
        return self.clock_loss(clock_output, clock_target)


    def forward(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates loss using the sum of loss values for specific elements of the baord 
        """
        out = (
            self.pices_loss_fn(output, target) 
            + self.piece_count_loss_fn(output)
            + self.turn_loss_fn(output, target)
            + self.castling_loss_fn(output, target)
            + self.en_passant_loss_fn(output, target)
            + self.clock_loss_fn(output, target)

        )
        return out




def load_train_test() -> tuple[list[str], str]:
    files = [f"board_tensorsV3/{DIR_2016_03}/{file}" for file in os.listdir(f"board_tensorsV3/{DIR_2016_03}")]
    test_file_name = "0_0.tnsrs"
    training_files = []
    test_file = ""
    for file in files:
        if test_file_name in file:
            test_file = file
        elif ".tnsrs" in file:
            training_files.append(file)
    return training_files, test_file





if __name__ == "__main__":
    training_files = []
    testing_files = []
    trainer = Trainer(
        training_files,
        testing_files,
        BoardAutoencoder(),
        "autoencoder",
        [TensorBoardUtilV4.SIZE],
        loss_fn_constructor = BoardLoss,
    )
    trainer.run()
