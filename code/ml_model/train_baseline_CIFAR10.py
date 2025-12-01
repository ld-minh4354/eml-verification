import os, sys
import argparse
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from resnet18_CIFAR10 import BasicBlock, ResNet



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If shape changes, use 1×1 conv to match dimensions for residual
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return F.relu(out)


class ResNet4(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 2 residual blocks
        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)  # downsample spatial size

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



class TrainBaselineCIFAR10:
    def __init__(self, seed = 10):
        self.add_project_folder_to_pythonpath()
        self.seed = seed
        self.set_seed(seed)
        self.device = torch.device("cuda")


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def main(self):
        self.load_data()
        self.set_hyperparameters()
        self.training()
        self.save_model()


    def load_data(self):
        self.num_classes = 10

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        os.makedirs("raw_datasets", exist_ok=True)
        train_dataset = datasets.CIFAR10(root="raw_datasets", train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10(root="raw_datasets", train=False, download=False, transform=transform_test)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


    def set_hyperparameters(self):
        self.EPOCH = 100
        self.LR = 1e-3
        self.WEIGHT_DECAY = 1e-4
        self.STEP_SIZE = 30
        self.GAMMA = 0.1


    def training(self):
        # self.model = models.resnet18()
        # self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.model.maxpool = nn.Identity()
        # self.model.avgpool = nn.AvgPool2d(kernel_size=4)
        # self.model.fc = nn.Linear(512, self.num_classes)
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_planes=16)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.STEP_SIZE, gamma=self.GAMMA)

        self.criterion = nn.CrossEntropyLoss()

        print(f"Start training ResNet18 for CIFAR10 under seed {self.seed}\n")

        for epoch in range(self.EPOCH):
            test_accuracy = self.train_loop(epoch)
            if test_accuracy >= 0.9:
                break

    
    def save_model(self):
        os.makedirs(os.path.join("models", "CIFAR10", "baseline"), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("models", "CIFAR10", "baseline", f"resnet18-CIFAR10-{self.seed}.pth"))

        x = torch.randn(1, 3, 32, 32).to(self.device)
        torch.onnx.export(self.model, x, os.path.join("models", "CIFAR10", "baseline", f"resnet18-CIFAR10-{self.seed}.onnx"),
                          export_params=True, external_data=False,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


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
        test_loss, test_accuracy = self.test_loop()

        self.scheduler.step()
        print(f"Epoch [{epoch+1:3d}] | Train Loss: {total_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        return test_accuracy
    

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

        return loss, accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10, help="Random seed for training")
    args = parser.parse_args()

    training = TrainBaselineCIFAR10(seed=args.seed)
    training.main()