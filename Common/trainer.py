from ast import Call
import torch
from torch import nn
import tqdm
from dataloader import load_tensor as dataloader_load_tensor
import os
from typing import Optional, Callable


class Trainer:

    BATCH_SCALE_TESTING = 64
    DEVICE = "mps"
    BASE_LEARNING_RATE = 1e-4

    def __init__(self, 
                 training_files : list[str], testing_file : str, 
                 model : nn.Module, model_name : str,
                 loss_fn_constructor : Callable[[torch.Tensor], nn.Module], in_datashape : list[int], 
                 batch_size : int = 64,
                 test_function : Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 model_save_name : Optional[str] = None,
                 device : str = "mps"):
       
        self.training_batch_size = batch_size
        self.testing_batch_size = batch_size * Trainer.BATCH_SCALE_TESTING
        self.datashape = in_datashape
        self.training_files = training_files
        self.test_tensor = self.load_tensor(testing_file, self.testing_batch_size)
        self.test_loss = loss_fn_constructor(self.test_tensor)
        self.model = model.to(Trainer.DEVICE)
        if model_name in os.listdir("models"):
            self.model.load_state_dict(torch.load(f"models/{model_name}", weights_only=True))
        self.model_save_name = model_name if model_save_name == None else model_save_name
        self.device = device

        
        self.loss_fn_constructor = loss_fn_constructor
        self.optimizer = torch.optim.SGD(self.model)
        self.test_function = test_function

        self.learning_rate = Trainer.BASE_LEARNING_RATE * (self.training_batch_size/8)


    def load_tensor(self, filename : str, batch_size : int) -> torch.Tensor:
        tensor = dataloader_load_tensor(filename).to(Trainer.DEVICE)
        found_datashape = list(tensor.shape[1:])
        assert(found_datashape == self.datashape)
        print(f"Loaded {filename} board tensor file of shape {tensor.shape}")
        batch_count = tensor.shape[0] // batch_size
        new_length = batch_count * batch_size
        tensor = tensor[:new_length]
        return torch.reshape(tensor, [batch_count, batch_size] + self.datashape)


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
        for batch in iterator:
            real = dataset[batch].to(Trainer.DEVICE)
            pred = self.model(real)
            loss : torch.Tensor = loss_fn(pred, real)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss_item = loss.item()
                average_loss = loss_item if iters == 0 else average_loss*(iters/(iters + 1)) + loss_item/iters
                iterator.set_description(f"Training loss: {loss_item}, Avg: {average_loss} ")
                iters += 1

    def test(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(self.test_tensor)), desc = "Testing"):
                real = self.test_tensor[i].to(Trainer.DEVICE)
                pred = self.model(real)
                total_loss += self.test_loss(pred, real).item()
            print(f"Average Testing Loss {total_loss/len(self.test_tensor)}")


    def save(self):
        torch.save(self.model.state_dict(),f"models/{self.model_save_name}")

    def run(self):

        for i in range(len(self.training_files)):
            self.test()
            training_tensor = self.load_training(i)
            self.train(training_tensor)
            self.save()
            


