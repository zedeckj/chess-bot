from ctypes.wintypes import LARGE_INTEGER
from chess import Board
import torch
from torch import Tensor, nn
import os

import tqdm
from board_final import TensorBoardUtilV4
from dataloader import LARGER_TENSORS, LARGER_TENSORS2, load_train_test, SMALL_TENSORS, TENSORS3
from eval_tools import precision_recall
import random
import chess

import unittest
from board_final import BoardGenerator
import itertools
from trainer import Trainer


MODEL_NAME = "128-autoencoder.pth"

DEVICE = "mps"

class BoardLoss(nn.Module):

    def _batch_data(self, dataset: torch.Tensor) -> torch.Tensor:
        BATCH_SIZE = 640
        current_length = dataset.shape[0]
        dataset = dataset[:current_length - (current_length % BATCH_SIZE)]
        dataset = torch.reshape(dataset, (dataset.shape[0] // BATCH_SIZE, BATCH_SIZE, dataset.shape[1]))
        return dataset

    def _generate_weights(self, dataset : torch.Tensor):

        with torch.no_grad():

            dataset = self._batch_data(dataset)

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
            print(bce_out.shape)
            ce_out = 1/(positives + 1e-14)
            pieces_weight = TensorBoardUtilV4.tensorToPieceTensors(ce_out)
            turn_weight = torch.squeeze(TensorBoardUtilV4.tensorToTurn(bce_out), dim = 0)
            castling_weight = torch.squeeze(TensorBoardUtilV4.tensorToCastlingRights(bce_out), dim = 0)
            castling_weight = torch.squeeze(TensorBoardUtilV4.tensorToCastlingRights(bce_out), dim = 0)
            castling_positives = TensorBoardUtilV4.tensorToCastlingRights(positives)
            print(f"Castling Counts: {castling_positives}")
            print(f"Castling Weight: {castling_weight}")
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


    def _piece_count_loss(self, piece_probabilities : torch.Tensor) :
        """
        Since there is never more than 16 pawns, and never more than 1 king for each color,
        we can add a loss term for predicting there are more than this number. We use the sum of probabilties
        to calculate the projected count.

        We also include bishops, rooks, and knights as having no more than 2.  
        """
        ...



    def pices_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        pieces_target = TensorBoardUtilV4.tensorToPieceTensors(target)
        piece_loss_list = []
        for i in range(64):
            class_label = torch.argmax(pieces_target[...,i, :], dim = 1)
            piece_loss_list.append(self.pieces_loss[i](pieces_output[...,i, :], class_label))
        return torch.stack(piece_loss_list).sum()
    
    def turn_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        turn_output = TensorBoardUtilV4.tensorToTurn(output)
        turn_target = TensorBoardUtilV4.tensorToTurn(target)
        loss = self.turn_loss(turn_output, turn_target)
        return loss

    def castling_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        castling_output = TensorBoardUtilV4.tensorToCastlingRights(output)
        castling_target = TensorBoardUtilV4.tensorToCastlingRights(target)
        loss = self.castling_loss(castling_output, castling_target)
        return loss
        
    def en_passant_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        en_passant_output = TensorBoardUtilV4.tensorToEnPassant(output)
        en_passant_target = torch.argmax(TensorBoardUtilV4.tensorToEnPassant(target), dim = 1)
        return self.en_passant_loss(en_passant_output, en_passant_target)
                                    
    def clock_loss_fn(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        clock_output = TensorBoardUtilV4.tensorToTimers(output)
        clock_target = TensorBoardUtilV4.tensorToTimers(target)
        return self.clock_loss(clock_output, clock_target)


    def forward(self, output : torch.Tensor, target : torch.Tensor):
        #turn_loss = self.turn_loss_fn(output, target)
        #castling_loss = self.castling_loss_fn(output, target)
        out = (
            self.pices_loss_fn(output, target) 
            + self.turn_loss_fn(output, target)
            + self.castling_loss_fn(output, target)
            + self.en_passant_loss_fn(output, target)
            + self.clock_loss_fn(output, target)
        )
        #print(f"TOTAL LOSS {out} CASTLING {castling_loss} TURN {turn_loss}")
        return out



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
    



if __name__ == "__main__":
    training_files = []
    testing_file = ""
    trainer = Trainer(
        training_files,
        testing_file,
        BoardAutoencoder(),
        "128-autoencoder.pth",
        BoardLoss,
        [1, TensorBoardUtilV4.SIZE]

    )
