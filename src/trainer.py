import torch
from torch import nn
from torch.optim import SGD
import os
import sys
from tqdm import tqdm
import pickle
from math import inf
from typing import Optional, Callable, Any
from dataloader import load_tensor as dataloader_load_tensor
import json

from src.board_final import TensorBoardUtilV4

class SelfSupervisedTrainer:
    DEVICE = "mps"
    MODEL_NAME = "deep-evaluator-mrl2"
    TRAINING_STR = "training"
    TESTING_STR = "testing"
    CURRENT_FILE_STR = "current_file"
    INITAL_LOSS = "initial_loss"

    def __init__(self, 
                 training_files : list[str], 
                 testing_files : list[str],
                 model : nn.Module,
                 model_name : str,
                 datashape : list[int],
                 preprocessor : Optional[nn.Module] = None,
                 loss_fn_constructor : Optional[Callable[[torch.Tensor], nn.Module]] = None,
                 test_callback : Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
                 loss_fn : Optional[nn.Module] = None,
                 base_learning_rate : float = 1e-4,
                 batch_size : int = 64,
                 device : str = "mps"):
        self.test_callback = test_callback
        self.device = device
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.learning_rate = base_learning_rate * (self.batch_size/8)
        self.training_files = training_files
        self.testing_files = testing_files
        self.datashape = datashape
        self.model = model
        self.model.to(device)
        self.model_name = model_name
        print(f"Batch size initialized to {self.batch_size}")
        if loss_fn_constructor != None:
            self.loss_fn = self.create_loss_fn(loss_fn_constructor)
        elif loss_fn != None:
            self.loss_fn = loss_fn
        else:
            raise Exception("Either loss_fn or loss_fn_constructor must be specified")
        self._init_directories()
        models = os.listdir("models")
        if f"{self.model_name}.pth" in models:
            self.model.load_state_dict(torch.load(f"models/{self.model_name}.pth", weights_only=True))
            print("Saved model loaded.")
        self.optimizier = SGD(self.model.parameters(), self.learning_rate)
        self.load_epoch_and_loss()

    def _init_directories(self):
        if "models" not in os.listdir("."):
            os.mkdir("models")
        if "losses" not in os.listdir("."):
            os.mkdir("losses")


    def create_loss_fn(self, constructor : Callable[[torch.Tensor], nn.Module]) -> nn.Module:
        MAX_FILE_COUNT = 5
        training_datasets = []
        i = 0
        while len(training_datasets) < min(MAX_FILE_COUNT, len(self.training_files)):
            data = self.load_training(i)
            if data != None:
                training_datasets.append(data)
            i += 1
        dataset = torch.cat(training_datasets, dim = 0)
        print("Constructing loss function...")
        return constructor(dataset)

    def train(self, dataset : torch.Tensor, sec : int, epoch: int, last_total : float):
        if last_total == None:
            print("Training file was previously corrupted, skipping")
            return
        self.model.train()
        iterable = tqdm(range(len(dataset)))
        total_loss = 0
        predictions = []
        for i in iterable:
            X = dataset[i]
            if self.preprocessor != None:
                X = self.preprocessor(X)
            pred = self.model(X)
            loss = self.loss_fn(pred, X)
            predictions.append(pred)
            total_loss += loss.item()
            loss.backward() 
            self.optimizier.step()
            self.optimizier.zero_grad()
            if i % 10 == 0 or len(dataset) - 1 == i:
                iterable.set_description(f"Epoch {epoch} section {sec} loss {total_loss:.4f}, prevous was {last_total:.4f}")
        if sec % 10 == 0 and sec != 0 and self.test_callback != None:
            self.test_callback(torch.cat(predictions), torch.reshape(dataset, (dataset.shape[0] * dataset.shape[1], dataset.shape[2])))
        return total_loss
        

    def test_single(self, dataset : torch.Tensor, section : int):
        self.model.eval()
        iteratable = tqdm(range(len(dataset)))
        total_loss = 0
        predictions = []
        with torch.no_grad():
            for i in iteratable:
                X = dataset[i]
                if self.preprocessor != None:
                    X = self.preprocessor(X)
                pred = self.model(X)
                predictions.append(pred)
                loss = self.loss_fn(pred, X)
                total_loss += loss.item()
                if i % 10 == 0 or len(dataset) - 1 == i:
                    iteratable.set_description(f"Testing section {section} loss {total_loss:.4f}")
        if self.test_callback != None:
            self.test_callback(torch.cat(predictions), torch.reshape(dataset, (dataset.shape[0] * dataset.shape[1], dataset.shape[2])))
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
        self.losses[SelfSupervisedTrainer.TESTING_STR].append(total_loss)
            


    def save(self):
        with open(f"losses/{self.model_name}.json", "w") as f:
            f.write(json.dumps(self.losses))
        torch.save(self.model.state_dict(),f"models/{self.model_name}.pth")
        print("Model and losses saved!")

    def load_epoch_and_loss(self):
        if f"{self.model_name}.json" in os.listdir("losses"):
            with open(f"losses/{self.model_name}.json", "r") as f:
                self.losses = json.loads(f.read())
                print("Loaded losses data")
            self.starting_epoch = len(self.losses[SelfSupervisedTrainer.TESTING_STR]) - 1
            return 
        self.starting_epoch = 0
        self.losses = {SelfSupervisedTrainer.TESTING_STR : [], SelfSupervisedTrainer.TRAINING_STR: [[]], SelfSupervisedTrainer.CURRENT_FILE_STR: 0}
                    


    def run(self):
        MAX_EPOCHS = 1000
        starting_index = self.losses[SelfSupervisedTrainer.CURRENT_FILE_STR] 
        if self.starting_epoch == 0 and starting_index == 0:
            self.run_test()
        for epoch in range(self.starting_epoch, MAX_EPOCHS):
            iterable = range(starting_index, len(self.training_files))
            for i in iterable:
                dataset = self.load_training(i)
                if dataset == None:
                    print("Invalid dataset, skipping")
                    self.losses[SelfSupervisedTrainer.TRAINING_STR][-1].append(None)
                    continue
                last_loss = inf if epoch == 0 else self.losses[SelfSupervisedTrainer.TRAINING_STR][epoch - 1][i]
                loss = self.train(dataset, i, epoch, last_loss)
                self.losses[SelfSupervisedTrainer.TRAINING_STR][-1].append(loss)
                self.losses[SelfSupervisedTrainer.CURRENT_FILE_STR] = i + 1
                if i % 10 == 0 and i != 0:
                    self.save()
            self.run_test()
            testing_loss = self.losses[SelfSupervisedTrainer.TESTING_STR][-1]
            last_testing_loss = self.losses[SelfSupervisedTrainer.TESTING_STR][-2]
            self.losses[SelfSupervisedTrainer.TRAINING_STR].append([])
            self.losses[SelfSupervisedTrainer.CURRENT_FILE_STR] = 0
            starting_index = 0
            self.save()
            if testing_loss > last_testing_loss:
                print(f"Testing loss {testing_loss}, not improved from {last_testing_loss}. Quiting training.")
                return
            else:
                print(f"Finished epoch {epoch} with testing loss of {testing_loss}, improved from {last_testing_loss}. Continuing Training\n")
                
            



    def load_tensor(self, filename : str) -> torch.Tensor:
        tensor = dataloader_load_tensor(filename).to(self.device)
        found_datashape = list(tensor.shape[-len(self.datashape):])
        assert(found_datashape == self.datashape)

        batch_count = tensor.shape[0] // self.batch_size
        new_length = batch_count * self.batch_size
        tensor = tensor[:new_length]
        out = torch.reshape(tensor, [batch_count, self.batch_size] + self.datashape)
        return out

    def load_testing(self, i : int) -> Optional[torch.Tensor]:
        try:
            out = self.load_tensor(self.testing_files[i])
            return out
        except:
            return None

    def load_training(self, i : int) -> Optional[torch.Tensor]:
        try:
            out = self.load_tensor(self.training_files[i])
            print(f"Loaded training file {i}")
            return out
        except Exception as e:
            print(f"Failed to load training file {i}: {e}")
            return None