import os, sys
from dnnv.properties import *
import numpy as np

project_path = os.path.abspath("")
if project_path not in sys.path:
    sys.path.append(project_path)

N = Network("N")
x = Image(os.path.join("test_data", "CIFAR10", "inputs", "input_22.npy"))

epsilon = Parameter("epsilon", type=float, default=0.005)
true_class = 8

Forall(
    x_,  
    Implies(
        ((x - epsilon) < x_ < (x + epsilon)) & (0 < x_ < 1),
        np.argmax(N(x_)) == true_class,
    ),
)
