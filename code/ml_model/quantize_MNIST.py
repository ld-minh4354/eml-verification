import os, sys
import argparse
import random

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import prune
from torchvision import datasets, transforms, models



class QuantizeMNIST:
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
        self.load_model()
        self.quantization()
        self.save_model()


    def load_data(self):
        self.num_classes = 10

        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        os.makedirs("raw_datasets", exist_ok=True)
        train_dataset = datasets.MNIST(root="raw_datasets", train=True, download=False, transform=transform_train)
        test_dataset = datasets.MNIST(root="raw_datasets", train=False, download=False, transform=transform_test)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    def load_model(self):
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.avgpool = nn.AvgPool2d(kernel_size=4)
        self.model.fc = nn.Linear(512, self.num_classes)
        self.model = self.model.to(self.device)

        state_dict = torch.load(os.path.join("models", "MNIST", "baseline", f"resnet18-MNIST-{self.seed}.pth"), 
                                map_location=self.device)
        self.model.load_state_dict(state_dict)


    def quantization(self):
        pass


    def save_model(self):
        os.makedirs(os.path.join("models", "MNIST", f"prune_{self.prune_rate}"), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("models", "MNIST", f"prune_{self.prune_rate}", f"resnet18-MNIST-{self.seed}.pth"))

        x = torch.randn(1, 1, 28, 28).to(self.device)
        torch.onnx.export(self.model, x, os.path.join("models", "MNIST", f"prune_{self.prune_rate}", f"resnet18-MNIST-{self.seed}.onnx"),
                          export_params=True, external_data=False,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10, help="Random seed for training")
    args = parser.parse_args()

    qt = QuantizeMNIST(seed=args.seed)
    qt.main()
