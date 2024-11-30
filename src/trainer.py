import torch
from torch import nn
from torch.optim import SGD
import os
import sys
from tqdm import tqdm
import pickle
from math import inf
from typing import Optional, Callable
from dataloader import load_tensor as dataloader_load_tensor

class Trainer:
    BATCH_SIZE = 64
    DEVICE = "mps"
    LEARNING_RATE = 1e-4 * (BATCH_SIZE/8)
    MODEL_NAME = "deep-evaluator-mrl2"

    def __init__(self, 
                 training_files : list[str], 
                 testing_files : list[str],
                 model : nn.Module,
                 model_name : str,
                 datashape : list[int],
                 loss_fn_constructor : Optional[Callable[[torch.Tensor], nn.Module]] = None,
                 loss_fn : Optional[nn.Module] = None,
                 batch_size : int = 64,
                 device : str = "mps"):
        if loss_fn_constructor != None:
            self.loss_fn = self.create_loss_fn(loss_fn_constructor)
        elif loss_fn != None:
            self.loss_fn = loss_fn
        else:
            raise Exception("Either loss_fn or loss_fn_constructor must be specified")
        self._init_directories()
        self.device = device
        self.datashape = datashape
        self.batch_size = batch_size
        self.learning_rate = 1e-4 * (self.batch_size/8)
        self.model = model
        self.model.to(device)
        self.model_name = model_name
        if model_name in os.listdir("models"):
            self.model.load_state_dict(torch.load(f"models/{self.model_name}", weights_only=True))
        self.epoch_losses = []   

        self.optimizier = SGD(self.model.parameters(), self.learning_rate)
        self.training_files = training_files
        self.testing_files = testing_files

    def _init_directories(self):
        if "models" not in os.listdir("."):
            os.mkdir("models")
        if "losses" not in os.listdir("."):
            os.mkdir("losses")


    def create_loss_fn(self, constructor : Callable[[torch.Tensor], nn.Module]) -> nn.Module:
        training_datasets = []
        for i in range(len(self.training_files)):
            training_datasets.append(self.load_training(i))
        dataset = torch.cat(training_datasets, dim = 0)
        return constructor(dataset)

    def train(self, dataset : torch.Tensor, sec : int, epoch: int, last_total : float):
        self.model.train()
        iterable = tqdm(range(len(dataset)))
        total_loss = 0
        for i in iterable:
            X = dataset[i]
            predictions = self.model(X)
            loss = self.loss_fn(predictions, X)
            total_loss += loss
            loss.backward() 
            self.optimizier.step()
            self.optimizier.zero_grad()
            if i % 10 == 0:
                iterable.set_description(f"Epoch {epoch} section {sec} loss {total_loss:.4f}, prevous was {last_total:.4f}")
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
        pickle.dump(self.epoch_losses, open(f"losses/{self.model_name}", "wb"))
        torch.save(self.model.state_dict(),f"models/{self.model_name}")

    def load_epoch(self):
        if self.model_name in os.listdir("losses"):
            with open(f"losses/{self.model_name}", "rb") as f:
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



    def load_tensor(self, filename : str, batch_size : int) -> torch.Tensor:
        tensor = dataloader_load_tensor(filename).to(self.device)
        found_datashape = list(tensor.shape[-len(self.datashape):])
        assert(found_datashape == self.datashape)

        batch_count = tensor.shape[0] // batch_size
        new_length = batch_count * batch_size
        tensor = tensor[:new_length]
        out = torch.reshape(tensor, [batch_count, batch_size] + self.datashape)
        return out

    def load_testing(self, i : int) -> Optional[torch.Tensor]:
        try:
            out = self.load_tensor(self.testing_files[i], self.batch_size)
            return out
        except:
            return None

    def load_training(self, i : int) -> Optional[torch.Tensor]:
        try:
            out = self.load_tensor(self.training_files[i], self.batch_size)
            return out
        except:
            return None