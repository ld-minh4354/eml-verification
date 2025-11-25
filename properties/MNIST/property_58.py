import os
from dnnv.properties import *
import numpy as np

N = Network("N")
x = Image(os.path.join("test_data", "MNIST", "inputs", "input_58.npy"))

epsilon = Parameter("epsilon", type=float)
true_class = 5

Forall(
    x_,  
    Implies(
        ((x - epsilon) < x_ < (x + epsilon)) & (0 < x_ < 1),
        np.argmax(N(x_)) == true_class,
    ),
)
