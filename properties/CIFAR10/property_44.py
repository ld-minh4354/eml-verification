import os
from dnnv.properties import *
import numpy as np

N = Network("N")
x = Image(os.path.join("test_data", "CIFAR10", "inputs", "input_44.npy"))

epsilon = Parameter("epsilon", type=float)
true_class = 6

Forall(
    x_,  
    Implies(
        ((x - epsilon) < x_ < (x + epsilon)) & (0 < x_ < 1),
        np.argmax(N(x_)) == true_class,
    ),
)
