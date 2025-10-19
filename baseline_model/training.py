import os, sys
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt



class TrainBaselineResNet:
    def __init__(self, dataset = "CIFAR10", seed = 10):
        self.add_project_folder_to_pythonpath()
        self.dataset = dataset
        self.seed = seed
        self.set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


    def main(self):
        self.load_data()
        self.set_hyperparameters()
        self.training()


    def load_data(self):
        if self.dataset == "CIFAR10":
            train_dataset = datasets.CIFAR10(
                root = os.path.join("data", "raw"),
                train = True,
                download = True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean = [0.4914, 0.4822, 0.4465],
                        std = [0.2023, 0.1994, 0.2010]
                    )
                ])
            )

            test_dataset = datasets.CIFAR10(
                root = os.path.join("data", "raw"),
                train = False,
                download = True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean = [0.4914, 0.4822, 0.4465],
                        std = [0.2023, 0.1994, 0.2010]
                    )
                ])
            )
        else:
            raise Exception("Not implemented yet.")

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


    def set_hyperparameters(self):
        self.EPOCH = 10
        self.LR = 0.001


    def training(self):
        self.model = models.resnet18().to(self.device)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        self.criterion = nn.CrossEntropyLoss()

        train_losses = []

        print(f"Start training ResNet18 for {self.dataset} under seed {self.seed}\n")

        for epoch in range(self.EPOCH):
            train_loss = self.train_loop(epoch)
            train_losses.append(train_loss)

        self.test_loop()

        torch.save(self.model.state_dict(), os.path.join("data", "model", f"resnet18-{self.dataset}-{self.seed}.pth"))


    def train_loop(self, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            self.optimizer.zero_grad()

            loss = self.criterion(outputs, targets)
            loss.backward()
            total_loss += loss.item()

            self.optimizer.step()

        total_loss = total_loss / len(self.train_loader)
        print(f"Epoch [{epoch+1:3d}] | Train Loss: {total_loss:.4f}")

        return total_loss
    

    def test_loop(self):
        self.model.eval()

        loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss += self.criterion(outputs, targets).item()

                pred = outputs.argmax(dim=1, keepdim=True)
                accuracy += pred.eq(targets.view_as(pred)).sum().item()

        loss = loss / len(self.test_loader)
        accuracy = accuracy / len(self.test_loader.dataset)

        print(f"Final Test Loss: {loss}")
        print(f"Final Test Accuracy: {accuracy}")



if __name__ == "__main__":
    training = TrainBaselineResNet(seed = 10)
    training.main()
        