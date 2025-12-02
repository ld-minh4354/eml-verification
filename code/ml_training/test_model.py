import numpy as np
import os

arr = np.load(os.path.join("test_data", "MNIST", "inputs", "input_3.npy"))
print(arr)
