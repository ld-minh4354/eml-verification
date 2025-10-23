import os, sys
import random

import pandas as pd
import numpy as np

import torch
from torchvision import datasets, transforms



class GeneratePropertyScripts:
    def __init__(self, dataset = "CIFAR10", epsilon = 0.005):
        self.add_project_folder_to_pythonpath()
        self.dataset = dataset
        self.epsilon = epsilon


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def get_file_content(self, index):
        return f"""
        import os, sys
        from dnnv.properties import *
        import numpy as np

        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)

        N = Network("N")
        x = Image(os.path.join())

        epsilon = Parameter("epsilon", type=float, default=(2.0 / 255))
        true_class = 7

        Forall(
            x_,  # x_ is assumed to be normalized, so denormalize before comparing to x
            Implies(
                ((x - epsilon) < denormalize(x_) < (x + epsilon)) & (0 < denormalize(x_) < 1),
                np.argmax(N(x_)) == true_class,
            ),
        )
        """
            

