from ctypes.wintypes import LARGE_INTEGER
from chess import Board
import torch
from torch import Tensor, nn
import os
import sys
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


    def _piece_count_loss(self, output : torch.Tensor) :
        """
        Since there is never more than 16 pawns, and never more than 1 king for each color,
        we can add a loss term for predicting there are more than this number. We use the sum of probabilties
        to calculate the projected count.

        We also include bishops, rooks, and knights as having no more than 2. 
        Altough, this is innacuarate due to the existence of underpromotion. An increase in 

        """
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        piece_probabilities = torch.zeros([13])



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



class AutoencoderTrainer:

    BATCH_SCALE_TESTING = 64
    DEVICE = "mps"
    BASE_LEARNING_RATE = 1e-4

    def __init__(self, 
                 training_files : list[str], testing_file : str, 
                 model : nn.Module, model_name : str,
                 loss_fn_constructor : Callable[[torch.Tensor], nn.Module], in_datashape : list[int], 
                 batch_size : int = 64,
                 model_save_name : Optional[str] = None,
                 device : str = "mps"):
       
        self.training_batch_size = batch_size
        self.testing_batch_size = batch_size * AutoencoderTrainer.BATCH_SCALE_TESTING
        self.datashape = in_datashape
        self.training_files = training_files
        self.test_tensor = self.load_tensor(testing_file, self.testing_batch_size)
        self.test_loss = loss_fn_constructor(self.test_tensor)
        self.model = model.to(AutoencoderTrainer.DEVICE)
        if model_name in os.listdir("models"):
            self.model.load_state_dict(torch.load(f"models/{model_name}", weights_only=True))
        self.model_save_name = model_name if model_save_name == None else model_save_name
        self.device = device

        
        self.loss_fn_constructor = loss_fn_constructor
        self.optimizer = torch.optim.SGD(self.model.parameters())

        self.learning_rate = AutoencoderTrainer.BASE_LEARNING_RATE * (self.training_batch_size/8)


    def load_tensor(self, filename : str, batch_size : int) -> torch.Tensor:
        tensor = dataloader_load_tensor(filename).to(AutoencoderTrainer.DEVICE)
        found_datashape = list(tensor.shape[-len(self.datashape):])
        print(f"Loaded {filename} board tensor file of shape {tensor.shape}")
        assert(found_datashape == self.datashape)

        batch_count = tensor.shape[0] // batch_size
        new_length = batch_count * batch_size
        tensor = tensor[:new_length]
        out = torch.reshape(tensor, [batch_count, batch_size] + self.datashape)
        print(f"Transformed into {out.shape}")
        return out


    def load_training(self, i : int) -> torch.Tensor:
        return self.load_tensor(self.training_files[i], self.training_batch_size)

    def shuffle(self, tensor : torch.Tensor):
        indices = torch.randperm(tensor.size(0))
        tensor = tensor[indices]
        return tensor

    def train(self, dataset : torch.Tensor):
        self.model.train()

        iterator = tqdm.tqdm(range(len(dataset)))
        dataset = self.shuffle(dataset)
        iters = 0
        average_loss = 0
        loss_fn = self.loss_fn_constructor(dataset)
        total_loss = 0
        pred_list = []
        for batch in iterator:
            real = dataset[batch].to(AutoencoderTrainer.DEVICE)
            pred = self.model(real)
            loss : torch.Tensor = loss_fn(pred, real)
            total_loss += loss.item()
            pred_list.append(pred)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss_item = loss.item()
                iterator.set_description(f"Training loss: {loss_item}, Avg: {total_loss / (batch + 1)} ")
                iters += 1
        display_precision_recall(torch.cat(pred_list), unbatch(dataset))


    def test(self):
        self.model.eval()
        total_loss = 0
        pred_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(self.test_tensor)), desc = "Testing"):
                real = self.test_tensor[i].to(AutoencoderTrainer.DEVICE)
                pred = self.model(real)
                pred_list.append(pred)
                total_loss += self.test_loss(pred, real).item()
            print(f"Average Testing Loss {total_loss/len(self.test_tensor)}")
            display_precision_recall(torch.cat(pred_list), unbatch(self.test_tensor))

    def save(self):
        torch.save(self.model.state_dict(),f"models/{self.model_save_name}")

    def run(self):
        for i in range(len(self.training_files)):
            self.test()
            training_tensor = self.load_training(i)
            self.train(training_tensor)
            self.save()
            print("saved!")
            print(f"Finished Epoch {i}\n")



if __name__ == "__main__":
    training_files, testing_file = load_train_test()
    trainer = AutoencoderTrainer(
        training_files,
        testing_file,
        BoardAutoencoder(),
        "128-autoencoder.pth",
        BoardLoss,
        [TensorBoardUtilV4.SIZE]
    )
    trainer.run()
