from doctest import testfile
from math import inf
from typing import Optional
import torch
from torch import nn
from torch.optim import SGD
from tqdm import tqdm
import sys
sys.path.append("./")
from src.board_final import BoardGenerator
from src.chess_models import Evaluator, ProductionAutoencoder
from src.dataloader import load_tensor as dataloader_load_tensor
import pickle 
import chess

INF = 50000
class EvaluatorLossCE(nn.Module):

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, evaluations : torch.Tensor):
        target = torch.zeros((evaluations.shape[0], 1), device = "mps") 
        return self.ce_loss(evaluations[0], target)
    

class EvaluatorLossMRL(nn.Module):

    def __init__(self):
        super().__init__()
        self.mrl_loss = nn.MarginRankingLoss(margin = 10)


    def forward(self, evaluations : torch.Tensor):
        target = torch.ones((evaluations.shape[0], 1), device = "mps") 
        return self.mrl_loss(evaluations[:,0], evaluations[:,1], target)
    

class EvaluatorPretrainer:


    def __init__(self, model : nn.Module):
        self.model = model
        self.optimizer = SGD(self.model.parameters(), EvaluatorTrainer.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.encoder = ProductionAutoencoder()



    def _calculate_material(self, board : chess.Board) -> float: 
        material = 0
        for square in range(64):
            piece = board.piece_at(square)
            if piece != None:
                if piece.piece_type == chess.PAWN:
                    value = 1
                elif piece.piece_type == chess.BISHOP:
                    value = 3
                elif piece.piece_type == chess.KNIGHT:
                    value = 3
                elif piece.piece_type == chess.ROOK:
                    value = 5
                elif piece.piece_type == chess.QUEEN:
                    value = 9
                else:
                    value = 0 # kings are always present
                if piece.color == chess.WHITE:
                    material += value
                else:
                    material -= value
        return material


    def heuristic(self, board: chess.Board) -> float:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -INF
            return INF
        else:
            return self._calculate_material(board)
        
    def pretrain(self):
        self.model.train()
        EXAMPLES = 1000000
        i = 0
        total_loss = 0
        for board in BoardGenerator(EXAMPLES):
            tensor = self.encoder.encodeFromBoard(board)
            evaluation = self.heuristic(board)
            pred = self.model(tensor)
            loss = self.loss_fn(pred, evaluation)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
            if i % 10:
                ...
                
            i += 1

        

class EvaluatorTrainer:
    BATCH_SIZE = 64
    DEVICE = "mps"
    LEARNING_RATE = 1e-4 * (BATCH_SIZE/8)
    DATASHAPE = [2, ProductionAutoencoder.TARGET_SIZE]
    MODEL_NAME = "deep-evaluator-mrl2"

    def __init__(self, training_files : list[str], testing_files : list[str]):
        self.loss_fn = EvaluatorLossMRL()
        self.model = Evaluator()
        self.model.to(EvaluatorTrainer.DEVICE)
        if EvaluatorTrainer.MODEL_NAME in os.listdir("models"):
            self.model.load_state_dict(torch.load(f"models/{EvaluatorTrainer.MODEL_NAME}", weights_only=True))
        self.optimizier = SGD(self.model.parameters(), EvaluatorTrainer.LEARNING_RATE)
        self.training_files = training_files
        self.used_files = []
        self.epoch_losses = []
        self.testing_files = testing_files

    def load_used(self) -> list[str]:
        """
        Unused function for recalling used training files
        """
        files = os.listdir("evaluator_used")
        if EvaluatorTrainer.MODEL_NAME in files:
            with open(f"evaluator_used/{EvaluatorTrainer.MODEL_NAME}", "rb") as f:
                return pickle.load(f)
        return []

    def train(self, dataset : torch.Tensor, sec : int, epoch: int, last_total : float):
        self.model.train()
        iteratable = tqdm(range(len(dataset)))
        total_loss = 0
        for i in iteratable:
            X = dataset[i]
            evaluations = self.model(X)
            loss = self.loss_fn(evaluations)
            total_loss += loss
            loss.backward() 
            self.optimizier.step()
            self.optimizier.zero_grad()
            if i % 10 == 0:
                iteratable.set_description(f"Epoch {epoch} section {sec} loss {total_loss:.4f}, prevous was {last_total:.4f}")
        return total_loss
        

    def test_single(self, dataset : torch.Tensor, section : int):
        self.model.eval()
        iteratable = tqdm(range(len(dataset)))
        total_loss = 0
        with torch.no_grad():
            for i in iteratable:
                X = dataset[i]
                evaluations = self.model(X)
                loss = self.loss_fn(evaluations)
                total_loss += loss
                if i % 10 == 0:
                    iteratable.set_description(f"Testing section {section} loss {total_loss:.4f}")
        return total_loss

    def run_test(self):
        iterable = range(len(self.testing_files))
        total_loss = 0
        for i in iterable:
            dataset = self.load_testing(i)
            if dataset == None:
                print("Invalid dataset, skipping")
                continue
            total_loss += self.test_single(dataset, i)
        print(f"Testing total loss is {total_loss}")
        return total_loss
            


    def save(self):
        pickle.dump(self.epoch_losses, open(f"evaluator_losses/{EvaluatorTrainer.MODEL_NAME}", "wb"))
        torch.save(self.model.state_dict(),f"models/{EvaluatorTrainer.MODEL_NAME}")

    def load_epoch(self):
        if EvaluatorTrainer.MODEL_NAME in os.listdir("evaluator_losses"):
            with open(f"evaluator_losses/{EvaluatorTrainer.MODEL_NAME}", "rb") as f:
                self.epoch_losses = pickle.load(f)
            return len(self.epoch_losses)
        return 0

    def run(self):
        MAX_EPOCHS = 1000
        iterable = range(len(self.training_files))
        lass_testing_loss = self.run_test()
        for epoch in range(self.load_epoch(), MAX_EPOCHS):
            losses = []
            for i in iterable:
                dataset = self.load_training(i)
                if dataset == None:
                    print("Invalid dataset, skipping")
                    continue
                loss = self.train(dataset, i, epoch, inf if epoch == 0 else self.epoch_losses[epoch - 1][i])
                losses.append(loss)
            self.epoch_losses.append(losses)
            testing_loss = self.run_test()
            if testing_loss > lass_testing_loss:
                print(f"Testing loss {testing_loss}, not improved from {lass_testing_loss}. Model no longer improving, finishing training")
                return
                
            print(f"Finished epoch {epoch} with testing loss of {testing_loss}, improved from {lass_testing_loss}, saving!\n\n")
            lass_testing_loss = testing_loss
            self.save()


    def not_trained(self, i : int):
        return self.training_files[i] not in self.used_files


    def load_tensor(self, filename : str, batch_size : int) -> torch.Tensor:
        tensor = dataloader_load_tensor(filename).to(EvaluatorTrainer.DEVICE)
        found_datashape = list(tensor.shape[-len(EvaluatorTrainer.DATASHAPE):])
        assert(found_datashape == EvaluatorTrainer.DATASHAPE)

        batch_count = tensor.shape[0] // batch_size
        new_length = batch_count * batch_size
        tensor = tensor[:new_length]
        out = torch.reshape(tensor, [batch_count, batch_size] + EvaluatorTrainer.DATASHAPE)
        return out

    def load_testing(self, i : int) -> Optional[torch.Tensor]:
        try:
            out = self.load_tensor(self.testing_files[i], EvaluatorTrainer.BATCH_SIZE)
            self.used_files.append(self.testing_files[i])
            return out
        except:
            return None

    def load_training(self, i : int) -> Optional[torch.Tensor]:
        try:
            out = self.load_tensor(self.training_files[i], EvaluatorTrainer.BATCH_SIZE)
            self.used_files.append(self.training_files[i])
            return out
        except:
            return None
    
if __name__ == "__main__":
    import os
    dir = "move_tensors/masters1"
    files = [f"{dir}/{f}" for f in os.listdir(dir)]
    files.sort()
    #test_files = files[0:20]
    #training_files = files[20:]
    test_files = files[0:20]
    train_files = files[20:]
    trainer = EvaluatorTrainer(train_files, test_files)
    trainer.run()
    