import os, sys
import random

import pandas as pd
import numpy as np

import torch
from torchvision import datasets, transforms



class GenerateInput:
    def __init__(self, dataset = "CIFAR10"):
        self.add_project_folder_to_pythonpath()
        self.dataset = dataset
        self.seed = 42
        self.set_seed(42)


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
        self.get_indices()
        self.create_input()


    def load_data(self):
        if self.dataset == "CIFAR10":
            self.num_classes = 10

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])

            os.makedirs("raw_datasets", exist_ok=True)
            self.test_dataset = datasets.CIFAR10(root="raw_datasets", train=False, download=True, transform=transform_test)

        elif self.dataset == "MNIST":
            self.num_classes = 10

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            os.makedirs("raw_datasets", exist_ok=True)
            self.test_dataset = datasets.MNIST(root="raw_datasets", train=False, download=True, transform=transform_test)

        else:
            raise Exception("Not implemented yet.")
        

    def get_indices(self):
        dataset_length = len(self.test_dataset)
        self.indices = np.random.randint(1, dataset_length + 1, size=100)
        print("Indices chosen:", self.indices)

    
    def create_input(self):
        df = pd.DataFrame(columns=["index", "label"])
        os.makedirs(os.path.join("test_data", self.dataset, "inputs"), exist_ok=True)

        for i in range(100):
            image, label = self.test_dataset[self.indices[i]]

            image_np = image.numpy()
            np.save(os.path.join("test_data", self.dataset, "inputs", f"input_{i}.npy"), image_np)

            df.loc[len(df)] = [i, label]
        
        df.to_csv(os.path.join("test_data", self.dataset, "labels.csv"), index=False)


if __name__ == "__main__":
    gt = GenerateInput(dataset="MNIST")
    gt.main()