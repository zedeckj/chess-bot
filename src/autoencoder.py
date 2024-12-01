from ctypes.wintypes import LARGE_INTEGER
from chess import Board
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import os
import sys

sys.path.append("./")
from src.trainer import SelfSupervisedTrainer
import tqdm
from src.chess_models import BoardAutoencoder
from src.board_final import TensorBoardUtilV4
from src.dataloader import DATASET_2016_03, DIR_2016_03, LARGER_TENSORS, LARGER_TENSORS2, SMALL_TENSORS, TENSORS3, DIR_2013_12
from src.dataloader import load_tensor as dataloader_load_tensor
from src.torch_utils import unbatch
import random
import chess
import math

import unittest
from board_final import BoardGenerator
import itertools
from typing import Callable, Optional


DEVICE = "mps"

class BoardLoss(nn.Module):

    EXPECTED_PIECES_COUNT = torch.tensor([])
    PRINT_LOSSES = False

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
            ce_out = lengths/(positives + 1e-14)
            pieces_weight = TensorBoardUtilV4.tensorToPieceTensors(ce_out)
            # We preform median here, not mean, due to pawns not being allowed in some rows
            median_pieces_weight = torch.squeeze(torch.median(pieces_weight, dim = 1)[0])
            turn_weight = torch.squeeze(TensorBoardUtilV4.tensorToTurn(bce_out), dim = 0)
            castling_weight = torch.squeeze(TensorBoardUtilV4.tensorToCastlingRights(bce_out), dim = 0)
            en_passant_weight = torch.squeeze(TensorBoardUtilV4.tensorToEnPassant(ce_out), dim = 0)
            simple_pieces_loss = nn.CrossEntropyLoss(weight = median_pieces_weight)
            """
            # We previously used 64 distinct instances of CrossEntropyLoss for each square, with distinct
            # weights for each piece. This was very very slow to use however
            piece_losses = []
            for weight in pieces_weight[0]:
                piece_losses.append(nn.CrossEntropyLoss(weight = weight))
            """
            turn_loss = nn.BCEWithLogitsLoss(pos_weight = turn_weight)
            castling_loss = nn.BCEWithLogitsLoss(pos_weight = castling_weight)
            en_passant_loss = nn.CrossEntropyLoss(weight = en_passant_weight)

            return simple_pieces_loss, turn_loss, castling_loss, en_passant_loss       


    def __init__(self, train_dataset):
        super(BoardLoss, self).__init__()

        simple_piece_loss, turn_loss, castling_loss, en_passant_loss = self._generate_weights(train_dataset)
        self.simple_piece_loss = simple_piece_loss
        self.turn_loss = turn_loss
        self.castling_loss = castling_loss
        self.en_passant_loss = en_passant_loss
        self.clock_loss = nn.L1Loss()
        self.pieces_count_loss = nn.L1Loss()


    def _create_piece_count_target(self):
        """
        Creates the target values for the piece loss count component of the loss function. 


        targets = torch.zeros(13, device = DEVICE)
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.WHITE))] = 8
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.BLACK))] = 8

        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KING, chess.WHITE))] = 1
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KING, chess.BLACK))] = 1


        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KNIGHT, chess.WHITE))] = 2
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.KNIGHT, chess.BLACK))] = 2

        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.BISHOP, chess.WHITE))] = 2
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.BISHOP, chess.BLACK))] = 2


        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.ROOK, chess.WHITE))] = 2
        targets[TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.ROOK, chess.BLACK))] = 2

        
        targets[TensorBoardUtilV4.indexOfPiece(None)] = 62 # There must always be at least 2 Kings

        return targets
        """
        ...

        



    def illegal_pawn_loss_fn(self, pieces_output: torch.Tensor) -> torch.Tensor:
        back_squares = torch.cat((pieces_output[:,0:8,:], pieces_output[:,56:64,:]))
        back_squares = torch.reshape(back_squares, (2 * 64 * 8, 13))
        softmax_squares = torch.softmax(back_squares, dim = -1)
        white_pawn_probabilities = softmax_squares[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.WHITE))]
        black_pawn_probabilities = softmax_squares[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.BLACK))]
        loss = torch.sum(torch.cat((white_pawn_probabilities, black_pawn_probabilities)))
        if BoardLoss.PRINT_LOSSES:
            print(f"PAWN LOSS {loss}")
        return loss
    
    def piece_count_loss_fn(self, pieces_output : torch.Tensor, pieces_target : torch.Tensor) -> torch.Tensor:
        """
        This loss term accounts for overpredicting the total number of each type of piece on boards.
        """
        EXP = 1/2
        PIECE_TYPES = 13

        pieces_softmax = torch.softmax(pieces_output, dim = -1) 
        # This scaling method is used to elimate non used pieces from count 
        threshold = torch.full_like(pieces_softmax, 1/PIECE_TYPES) 
        pieces_scaled = (torch.clamp(pieces_softmax - threshold, min = 1e-8) ** EXP) * ((PIECE_TYPES/(PIECE_TYPES - 1)) ** EXP)

        output_summed = torch.sum(pieces_scaled, dim = 1) 
        target_summed = torch.sum(pieces_target, dim = 1)
        loss = torch.clamp(output_summed - target_summed, min = 0)
        # print(output_summed.shape)
        # print(f"original: {pieces_output[0,0,:]} processed: {output_summed[30]}\ntarget: {target_summed[30]}\nloss: {loss[30].tolist()}")
        loss = loss.mean()
        if BoardLoss.PRINT_LOSSES:
            print(f"PIECE COUNT {loss}")
        return loss





    def piece_invariants_loss_fn(self, output : torch.Tensor) -> torch.Tensor:
        """
        Since there is never more than 16 pawns, and never more than 1 king for each color,
        we can add a loss term for predicting there are more than this number. We use the sum of probabilties
        to calculate the projected count. 

        We also include bishops, rooks, and knights as having no more than 2. 
        Altough, this is innacuarate in edge cases due to the existence of underpromotion. However,
        the usage of under promotion while also having two other instances of the target piece is 
        too rare of an occurance to bother with, and likely never appears in the training set.

        In practice, this function caused the vast overrepresentation of queens


        SCALE = 16 # While this metric is important, weighting it too causes queens to be over predicted 
        MAX_WHITE_PIECES = 16
        MAX_BLACK_PIECES = 16

        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output).to(DEVICE) # [BATCH_SIZE, 64, 13]

        pieces_softmax = torch.softmax(pieces_output, dim = 2) # [BATCH_SIZE, 64, 13]
        pieces_summed = torch.sum(pieces_softmax, dim = 1) # [BATCH_SIZE, 13]

        # we expect a shape of [N, 13] at this point
        pieces_summed[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.QUEEN, chess.WHITE))] = 0
        pieces_summed[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.QUEEN, chess.BLACK))] = 0
        # queens excluded

        


        possible_loss = (pieces_summed - self.pieces_count_targets)
        print(pieces_summed[0])
        print(possible_loss[0])
        zeroes = torch.zeros(possible_loss.shape).to(DEVICE)
        loss = torch.maximum(possible_loss, zeroes).mean()
        # print(f"PIECES COUNT LOSS {loss}")
        return loss * SCALE
        """
        ...

    def piece_loss_fn(self, pieces_output : torch.Tensor, pieces_target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for the label of each type of piece at each square. This is done by reshaping the output 
        and target into tensors of shape BATCH_SIZE * 64, 13, where 64 is the number of square on a board, and 13 is the number of 
        piece classes.  
        """
        pieces_output = pieces_output.reshape((pieces_output.shape[0] * 64, 13))
        pieces_target = pieces_target.reshape((pieces_target.shape[0] * 64, 13))
        class_labels = torch.argmax(pieces_target, dim = 1)
        loss = self.simple_piece_loss(pieces_output, class_labels)
        if BoardLoss.PRINT_LOSSES:
            print(f"PIECE TYPE {loss}")
        return loss 
    
    """
    def pieces_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        # 
        # Calculates the loss value for piece placement, using 64 distinct CE losses. Revised slightly to parallize generating class labels. 
        # Since there exists a unique loss function for each piece-square, this part cannot be parallelized. 
        #
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        pieces_target = TensorBoardUtilV4.tensorToPieceTensors(target)
        class_labels = torch.argmax(pieces_target, dim = 2)
        piece_loss_list = []
        for i in range(64):
            # This loop is VERY costly. Should figure out how to parallelize in the future
            piece_loss_list.append(self.pieces_loss[i](pieces_output[...,i, :], class_labels[i]))
        
        loss = torch.stack(piece_loss_list).sum()
        # print(f"PIECES LOSS {loss}")
        return loss
    """


    def turn_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for whose turn it is, which is encoded as a single logit, using Binary Cross Entroy.
        """
        turn_output = TensorBoardUtilV4.tensorToTurn(output)
        turn_target = TensorBoardUtilV4.tensorToTurn(target)
        loss = self.turn_loss(turn_output, turn_target)
        if BoardLoss.PRINT_LOSSES:
            print(f"TURN LOSS {loss}")
        return loss 

    def castling_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for castling rights, using Binary Cross Entropy for the 4 castling bits.
        """
        castling_output = TensorBoardUtilV4.tensorToCastlingRights(output)
        castling_target = TensorBoardUtilV4.tensorToCastlingRights(target)
        loss = self.castling_loss(castling_output, castling_target)
        # print(f"CASTLING LOSS {loss}")
        return loss 
        
    def en_passant_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the en passant square, using Cross Entropy Loss over the 65 logits.
        """
        en_passant_output = TensorBoardUtilV4.tensorToEnPassant(output)
        en_passant_target = torch.argmax(TensorBoardUtilV4.tensorToEnPassant(target), dim = 1)
        loss = self.en_passant_loss(en_passant_output, en_passant_target)
        if BoardLoss.PRINT_LOSSES:
            print(f"EP LOSS {loss}")
        return loss 
                                    
    def clock_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the half-move and full-move clocks using L1 loss, or MAE.
        """
        clock_output = TensorBoardUtilV4.tensorToTimers(output)
        clock_target = TensorBoardUtilV4.tensorToTimers(target)
        loss = self.clock_loss(clock_output, clock_target) 
        if BoardLoss.PRINT_LOSSES:
            print(f"CLOCK LOSS {loss}")
        return loss 


    def forward(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates loss using the sum of loss values for specific elements of the baord 
        """
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        pieces_target = TensorBoardUtilV4.tensorToPieceTensors(target)
        out = (
            self.piece_loss_fn(pieces_output, pieces_target) 
            + self.piece_count_loss_fn(pieces_output, pieces_target)
            + self.illegal_pawn_loss_fn(pieces_output)
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
    PATH = "board_tensorsV3/2016-03"
    files = [f"{PATH}/{file}" for file in os.listdir(PATH) if ".tnsrs" in file]
    files.sort()
    testing_files = files[:5]
    training_files = files[5:]
    trainer = SelfSupervisedTrainer(
        training_files,
        testing_files,
        BoardAutoencoder(),
        "autoencoder",
        [TensorBoardUtilV4.SIZE],
        loss_fn_constructor = BoardLoss,
    )
    trainer.run()
