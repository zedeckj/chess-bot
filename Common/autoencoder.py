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

device = "mps"



BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 1e-4 * (BATCH_SIZE/8)

MODEL_NAME = "128-autoencoder.pth"

def increase_batch_size(dataset: torch.Tensor, scale : int) -> torch.Tensor:
    current_batches = dataset.shape[0]
    dataset = dataset[:current_batches - (current_batches % scale)]
    dataset = torch.reshape(dataset, (dataset.shape[0] // scale, BATCH_SIZE * scale, dataset.shape[2]))
    return dataset

class BoardLoss(nn.Module):


    def _generate_weights(self, dataset : torch.Tensor):
        NEW_BATCH_SCALE = 1000

        with torch.no_grad():

            dataset = increase_batch_size(dataset, NEW_BATCH_SCALE)

            positives = torch.zeros((1,dataset.shape[2]), device = device)
            bitmask = torch.ones(TensorBoardUtilV4.BINARY_RANGE, device = device)
            mask = torch.unsqueeze(torch.cat([bitmask, torch.zeros(2, device = device)]),0)
            for i in tqdm.tqdm(range(len(dataset)), desc = "Generating Loss Weights"):
                tensor = dataset[i]
                tensor = tensor.to(device)
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


    """
    def forward(self, output : torch.Tensor, target : torch.Tensor):
        pieces_output = TensorBoardUtilV4.tensorToPieceTensors(output)
        turn_output = TensorBoardUtilV4.tensorToTurn(output)
        castling_output = TensorBoardUtilV4.tensorToCastlingRights(output)
        en_passant_output = TensorBoardUtilV4.tensorToEnPassant(output)
        clock_output = TensorBoardUtilV4.tensorToTimers(output)


        pieces_target = TensorBoardUtilV4.tensorToPieceTensors(target)
        turn_target = TensorBoardUtilV4.tensorToTurn(target)
        castling_target = TensorBoardUtilV4.tensorToCastlingRights(target)
        en_passant_target = torch.argmax(TensorBoardUtilV4.tensorToEnPassant(target), dim = 1)
        clock_target = TensorBoardUtilV4.tensorToTimers(target)
        piece_loss_list = []
        for i in range(64):
            class_label = torch.argmax(pieces_target[...,i, :], dim = 1)
            piece_loss_list.append(self.pieces_loss[i](pieces_output[...,i, :], class_label))
        pieces_loss_val = torch.stack(piece_loss_list).sum()
        turn_loss_val = self.turn_loss(turn_output, turn_target)
        castling_loss_val = self.castling_loss(castling_output, castling_target)
        en_passant_loss_val = self.en_passant_loss(en_passant_output, en_passant_target)
        clock_loss_val = self.clock_loss(clock_output, clock_target)
        return pieces_loss_val + turn_loss_val + castling_loss_val + en_passant_loss_val + clock_loss_val
    """

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


class BoardAutoencoderV1(nn.Module):

    TARGET_SIZE = 256
    LAYER_A = 2048
    LAYER_B = 1024
    LAYER_C = 512

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(TensorBoardUtilV4.SIZE, BoardAutoencoderV1.LAYER_A),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV1.LAYER_A, BoardAutoencoderV1.LAYER_B),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV1.LAYER_B, BoardAutoencoderV1.LAYER_C),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV1.LAYER_C, BoardAutoencoderV1.TARGET_SIZE),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BoardAutoencoderV1.TARGET_SIZE, BoardAutoencoderV1.LAYER_C),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV1.LAYER_C, BoardAutoencoderV1.LAYER_B),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV1.LAYER_B, BoardAutoencoderV1.LAYER_A),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV1.LAYER_A, TensorBoardUtilV4.SIZE),
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
    

class BoardAutoencoderV2(BoardAutoencoderV1):
    LAYER_A = 2048
    LAYER_B = 1024
    LAYER_C = 512
    TARGET_SIZE = 128

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(TensorBoardUtilV4.SIZE, BoardAutoencoderV2.LAYER_A),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV2.LAYER_A, BoardAutoencoderV2.LAYER_B),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV2.LAYER_B, BoardAutoencoderV2.LAYER_C),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV2.LAYER_C, BoardAutoencoderV2.TARGET_SIZE),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BoardAutoencoderV2.TARGET_SIZE, BoardAutoencoderV2.LAYER_C),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV2.LAYER_C, BoardAutoencoderV2.LAYER_B),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV2.LAYER_B, BoardAutoencoderV2.LAYER_A),
            nn.ReLU(),
            nn.Linear(BoardAutoencoderV2.LAYER_A, TensorBoardUtilV4.SIZE),
        )
    


def train(dataset : torch.Tensor, model : BoardAutoencoderV2, loss_fn : nn.Module, optimizer):

    model.train()
    iterator = tqdm.tqdm(range(len(dataset)))
    average_loss = 0
    iters = 0
    indices = torch.randperm(dataset.size(0))
    dataset = dataset[indices]
    
    for batch in iterator:
        real = dataset[batch].to(device)

        pred = model(real)
        loss = loss_fn(pred, real)

        loss.backward()
        for _, param in model.named_parameters():
            assert(param.grad is not None)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_item = loss.item()
            average_loss = loss_item if iters == 0 else average_loss*(iters/(iters + 1)) + loss_item/iters
            iterator.set_description(f"Training loss: {loss_item}, Avg: {average_loss} ")
            iters += 1

def test(dataset : torch.Tensor, loss_fn : nn.Module, model : BoardAutoencoderV2):
    model.eval()
    acc_sum = torch.zeros(13, device= device)
    rec_sum = torch.zeros(13, device= device)
    numerator = torch.tensor(0)
    total_loss = 0
    with torch.no_grad():
        dataset = increase_batch_size(dataset, 1000)
        for i in tqdm.tqdm(range(len(dataset)), desc = "testing"):
            real = dataset[i].to(device)
            pred = model(real)
            total_loss += loss_fn(pred, real).item()
            discrete_output = TensorBoardUtilV4.discretizeTensor(pred)
            discrete_target = real
            acc, rec = precision_recall(discrete_output, discrete_target)
            acc_sum += acc
            rec_sum += rec
            numerator += 1
    acc = acc_sum / numerator
    rec = rec_sum / numerator
    index_to_name = ["White Pawn", "White Knight", "White Bishop", "White Rook", "White Queen", "White King", 
                     "Black Pawn", "Black Knight", "Black Bishop", "Black Rook", "Black Queen", "Black King",
                     "Black Space"]
    for i in range(13):
        print(f"{index_to_name[i]} Precision: {acc[0].item()*100}%")
        print(f"{index_to_name[i]} Recall: {rec[0].item() * 100}%")
    print(f"Avg Testing Loss {total_loss/(numerator)}")


def run(train_dataset, test_dataset, model):
    loss_fn = BoardLoss(train_dataset)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for i in range(EPOCHS):
        print(f"On epoch {i}")
        test(test_dataset, loss_fn, model)
        train(train_dataset, model, loss_fn, optimizer)
        torch.save(model.state_dict(),f"models/{MODEL_NAME}")
        print("saved!")



SHOULD_TEST = False
if SHOULD_TEST:
    _, DATASET_TESTING = load_train_test(TENSORS3, 16, 0.5)
    TEST_MODEL = BoardAutoencoderV2()
    if MODEL_NAME in os.listdir("models"):
            TEST_MODEL.load_state_dict(torch.load(f"models/{MODEL_NAME}", weights_only=True))
    TEST_LOSS_FN = BoardLoss(train_dataset= DATASET_TESTING)

class TestAutoencoder(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName = methodName)
        self.test_dataset = DATASET_TESTING
        self.loss_fn = TEST_LOSS_FN
        self.model = TEST_MODEL


    def testEnPassantLoss(self):
        ep_squares = [None] + list(range(64))
        for square in ep_squares:
            board = chess.Board()
            board.ep_square = square
            tensor = TensorBoardUtilV4.fromBoard(board).to("mps")
            true_loss = self.loss_fn.en_passant_loss_fn(tensor, tensor).item()
            losses = {}
            for s2 in ep_squares:
                board2 = chess.Board()
                board2.ep_square = s2
                tensor2 = TensorBoardUtilV4.fromBoard(board2).to("mps")
                losses[s2] = (self.loss_fn.en_passant_loss_fn(tensor, tensor2).item())
            min_loss = min(losses.values())
            self.assertEqual(min_loss, true_loss)
            for s2 in ep_squares:
                if s2 != square:
                    self.assertGreater(losses[s2], true_loss)

    def testModelParamaters(self):
        model = BoardAutoencoderV2()
        if MODEL_NAME in os.listdir("models"):
            model.load_state_dict(torch.load(f"models/{MODEL_NAME}", weights_only=True))
        for value in model.encoder.parameters():
            self.assertEqual(torch.sum(torch.eq(value, torch.zeros_like(value)).float()), 0)
        for value in model.decoder.parameters():
            self.assertEqual(torch.sum(torch.eq(value, torch.zeros_like(value)).float()), 0)



    def testCastlingLoss(self):
        #this whole method was wrong
        """
        full_cs = "KQkq"
        castle_strings =  [''.join(tup) for tup in (list(itertools.combinations(full_cs, 1)) 
                    + list(itertools.combinations(full_cs, 2)) 
                    + list(itertools.combinations(full_cs, 3)) 
                    + list(itertools.combinations(full_cs, 4)))]
        for c in castle_strings:
            board = chess.Board()
            board.set_castling_fen(c)
            tensor = TensorBoardUtilV4.fromBoard(board).to("mps")
            true_loss = self.loss_fn.en_passant_loss_fn(tensor, tensor).item() # what???
            losses = {}
            for c2 in castle_strings:
                board2 = chess.Board()
                board2.set_castling_fen(c2)
                tensor2 = TensorBoardUtilV4.fromBoard(board2).to("mps")
                losses[c2] = (self.loss_fn.en_passant_loss_fn(tensor, tensor2).item())
            min_loss = min(losses.values())
            self.assertEqual(min_loss, true_loss)
            for c2 in castle_strings:
                if c != c2:
                    self.assertGreater(losses[c2], true_loss)
        """
        

    def test(self):
        self.assertEqual(True, True)


def main():
    train_dataset, test_dataset = load_train_test(TENSORS3, BATCH_SIZE, 1)
    model = BoardAutoencoderV2().to(device)
    if MODEL_NAME in os.listdir("models"):
        model.load_state_dict(torch.load(f"models/{MODEL_NAME}", weights_only=True))
        print(f"{MODEL_NAME} loaded!")
    run(train_dataset, test_dataset, model)

def viewTurnLoss(loss_fn : BoardLoss):
    for board in BoardGenerator(10):
        tensor1 = TensorBoardUtilV4.fromBoard(board).to(device)
        if board.turn == chess.WHITE:
            board.turn = chess.BLACK
        else:
            board.turn = chess.WHITE
        tensor2 = TensorBoardUtilV4.fromBoard(board).to(device)
        loss1 = loss_fn.turn_loss_fn(tensor1, tensor1)
        loss2 = loss_fn.turn_loss_fn(tensor1, tensor2)
        print(f"Correct loss: {loss1} Incorrect: {loss2}")


if __name__ == "__main__":
    if SHOULD_TEST:
        #unittest.main()
        viewTurnLoss(TEST_LOSS_FN) # type:ignore
    else:
        main()




