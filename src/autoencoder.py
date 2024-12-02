from ctypes.wintypes import LARGE_INTEGER
from turtle import back
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

from board_final import BoardGenerator
from typing import Callable, Optional


DEVICE = "mps"

class BoardLoss(nn.Module):

    EXPECTED_PIECES_COUNT = torch.tensor([])
    PRINT_LOSSES = False

    def _generate_weights(self, dataset : torch.Tensor):
        with torch.no_grad():


            positives = torch.zeros((1,TensorBoardUtilV4.SIZE), device = DEVICE)
            bitmask = torch.ones(TensorBoardUtilV4.BINARY_RANGE, device = DEVICE)
            mask = torch.unsqueeze(torch.cat([bitmask, torch.zeros(2, device = DEVICE)]),0)
            for i in tqdm.tqdm(range(len(dataset)), desc = "Generating Loss Weights"):
                tensor = dataset[i]
                tensor = tensor.to(DEVICE)
                tensor = mask * tensor
                positives += torch.sum(tensor, dim = 0) 
            lengths = dataset.shape[0] * dataset.shape[1]
            full_pieces_positives = torch.squeeze(TensorBoardUtilV4.tensorToPieceTensors(positives))
            self.ep_illegal_mask = torch.lt(TensorBoardUtilV4.tensorToEnPassant(positives), 1).float()
            bce_out = (lengths - positives)/(positives + 1e-8)
            ce_out = lengths/(positives + 1e-14)
            #pieces_weight = torch.squeeze(lengths/pieces_positives)
            #pieces_weight = lengths/(full_pieces_positives + 1e-14)
            NORMALIZE_EXP = 1/2 # used to reduce extreme variance, which cause unstable loss evaluations
            pieces_weight = (lengths ** NORMALIZE_EXP)/((full_pieces_positives + 1e-14) ** NORMALIZE_EXP) 
            """
            weights_list = torch.flatten(pieces_weight).tolist()
            weights_list.sort()
            print(weights_list)
            assert(False)
            """
            #print(pieces_weight.median(dim = 1))
            self.full_pieces_weight = torch.cat([pieces_weight for _ in range(dataset.shape[1])])
            turn_weight = torch.squeeze(TensorBoardUtilV4.tensorToTurn(bce_out), dim = 0)
            castling_weight = torch.squeeze(TensorBoardUtilV4.tensorToCastlingRights(bce_out), dim = 0)
            en_passant_weight = torch.squeeze(TensorBoardUtilV4.tensorToEnPassant(ce_out), dim = 0)
            #simple_pieces_loss = nn.CrossEntropyLoss(weight = pieces_weight)
            bare_pieces_loss = nn.CrossEntropyLoss(reduction = "none")
            """
            # We previously used 64 distinct instances of CrossEntropyLoss for each square, with distinct
            # weights for each piece. This was very very slow to use
            piece_losses = []
            for weight in pieces_weight[0]:
                piece_losses.append(nn.CrossEntropyLoss(weight = weight))
            """
            turn_loss = nn.BCEWithLogitsLoss(pos_weight = turn_weight)
            castling_loss = nn.BCEWithLogitsLoss(pos_weight = castling_weight)
            en_passant_loss = nn.CrossEntropyLoss(weight = en_passant_weight)

            return bare_pieces_loss, turn_loss, castling_loss, en_passant_loss       


    def __init__(self, train_dataset):
        super(BoardLoss, self).__init__()

        bare_pieces_loss, turn_loss, castling_loss, en_passant_loss = self._generate_weights(train_dataset)
        self.bare_pieces_loss = bare_pieces_loss
        self.turn_loss = turn_loss
        self.castling_loss = castling_loss
        self.en_passant_loss = en_passant_loss
        self.l1_loss = nn.L1Loss()

    def piece_count_loss(self, pieces_output: torch.Tensor, pieces_target : torch.Tensor) -> torch.Tensor:
        """
        Generates a loss value for differences between the summed probabilities of pieces_output and the true counts
        of each piece
        """
        PIECE_COUNT = 13
        # excluding empty squares, as with sparse boards they dominate the loss here
        pieces_output = pieces_output[:,:,0:12]
        pieces_target = pieces_target[:,:,0:12]
        softmax_output = torch.softmax(pieces_output, dim = -1)
        # rescaled to exclude probabilities of pieces that are certainly not the maximum
        rescaled_ouput = torch.clamp((softmax_output - 1/PIECE_COUNT), min = 0) * PIECE_COUNT/(PIECE_COUNT - 1)  
        summed_output = torch.sum(rescaled_ouput, dim = 1)
        summed_target = torch.sum(pieces_target, dim = 1)
        loss = self.l1_loss(summed_output, summed_target)
        if BoardLoss.PRINT_LOSSES:
            print(f"COUNT LOSS {loss}")
        return loss
    

    def illegal_pawn_loss_fn(self, pieces_output: torch.Tensor) -> torch.Tensor:
        """
        Pawns cannot exist on rows 1 and 8. The model should be punished heavily for predicting any non zero
        probability of a pawn being in these rows
        """
        back_squares = torch.cat((pieces_output[:,0:8,:], pieces_output[:,56:64,:]))
        back_squares = torch.reshape(back_squares, (2 * pieces_output.shape[0] * 8, 13))
        softmax_squares = torch.softmax(back_squares, dim = -1)
        white_pawn_probabilities = softmax_squares[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.WHITE))]
        black_pawn_probabilities = softmax_squares[:,TensorBoardUtilV4.indexOfPiece(chess.Piece(chess.PAWN, chess.BLACK))]
        loss = torch.mean(torch.cat((white_pawn_probabilities, black_pawn_probabilities)))
        if BoardLoss.PRINT_LOSSES:
            print(f"PAWN LOSS {loss}")
        return loss
    



    def piece_loss_fn(self, pieces_output : torch.Tensor, pieces_target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for the label of each type of piece at each square. This is done by reshaping the output 
        and target into tensors of shape BATCH_SIZE * 64, 13, where 64 is the number of square on a board, and 13 is the number of 
        piece classes.  
        """
        pieces_output = torch.reshape(pieces_output, (pieces_output.shape[0] * 64, 13))
        pieces_target = torch.reshape(pieces_target, (pieces_target.shape[0] * 64, 13))
        class_labels = torch.argmax(pieces_target, dim = -1)
        loss_unreduced = self.bare_pieces_loss(pieces_output, class_labels) 
        weights = torch.sum(pieces_target * self.full_pieces_weight, dim = -1)
        
        loss = torch.mean(loss_unreduced * weights)
        if BoardLoss.PRINT_LOSSES:
            #i = torch.argmax(loss_unreduced)
            #print(f"pieces_output: {pieces_output[i]}\n pieces_target: {pieces_target[i]}\n loss_unr = {loss_unreduced[i]}\n weights = {weights[i]}\n full_weights {self.full_pieces_weight[i]}\nloss= {loss}")
            print(f"PIECE TYPE LOSS {loss}")
        return loss 


    def turn_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for whose turn it is, which is encoded as a single logit, using Binary Cross Entroy.
        """
        turn_output = TensorBoardUtilV4.tensorToTurn(output)
        turn_target = TensorBoardUtilV4.tensorToTurn(target)
        loss = self.turn_loss(turn_output + 1e-14, turn_target)
        if BoardLoss.PRINT_LOSSES:
            print(f"TURN LOSS {loss}")
        return loss 

    def castling_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss value for castling rights, using Binary Cross Entropy for the 4 castling bits.
        """
        castling_output = TensorBoardUtilV4.tensorToCastlingRights(output)
        castling_target = TensorBoardUtilV4.tensorToCastlingRights(target)
        loss = self.castling_loss(castling_output + 1e-14, castling_target)
        if BoardLoss.PRINT_LOSSES:
            print(f"CASTLING LOSS {loss}")
        return loss 
        
    def en_passant_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the en passant square, using Cross Entropy Loss over the 65 logits.
        """
        en_passant_output = TensorBoardUtilV4.tensorToEnPassant(output)
        en_passant_target = torch.argmax(TensorBoardUtilV4.tensorToEnPassant(target), dim = 1)
        loss = self.en_passant_loss(en_passant_output, en_passant_target)
        en_passant_softmax = torch.softmax(en_passant_output, dim = -1)
        illegal_en_passant = en_passant_softmax * self.ep_illegal_mask
        illegal_en_passant_loss = torch.mean(illegal_en_passant)
        if BoardLoss.PRINT_LOSSES:
            print(f"EP LOSS {loss}")
            print(f"ILLEGAL EP LOSS {illegal_en_passant_loss}")
        return loss + illegal_en_passant_loss
                                    
    def clock_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the half-move and full-move clocks using L1 loss, or MAE.
        """
        clock_output = TensorBoardUtilV4.tensorToTimers(output)
        clock_target = TensorBoardUtilV4.tensorToTimers(target)
        loss = self.l1_loss(clock_output, clock_target) 
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
            + self.piece_count_loss(pieces_output, pieces_target)
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
        batch_size= 2048,
        base_learning_rate= 1e-4
    )
    trainer.run()
