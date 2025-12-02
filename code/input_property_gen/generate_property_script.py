import os, sys
import textwrap
import itertools

import pandas as pd



class GeneratePropertyScripts:
    def __init__(self, epsilon=0.001):
        self.add_project_folder_to_pythonpath()
        self.epsilon = str(epsilon)

        self.model_types = ["baseline",
                            "prune_0.2", "prune_0.4", "prune_0.6", "prune_0.7",
                            "prune_0.75", "prune_0.8", "prune_0.85", "prune_0.9"]
        self.seed_values = list(range(10, 101, 10))
        self.property_values = list(range(100))


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def main(self):
        os.makedirs(os.path.join("properties", "MNIST", self.epsilon), exist_ok=True)
        index = 0

        for model, seed, property in itertools.product(self.model_types, self.seed_values, self.property_values):
            file_content = self.get_file_content(model, seed, property)

            file_path = os.path.join("properties", "MNIST", self.epsilon, f"{index}.yaml")
            with open(file_path, "w") as f:
                f.write(file_content)

            index += 1


    def get_file_content(self, model, seed, property):
        return textwrap.dedent(f"""\
            model:
                name: eml_mnist
                path: models/MNIST/{model}/resnet4-MNIST-{seed}.pth
            data:
                dataset: MNIST
                mean: [0.1307]
                std:  [0.3081]
                start: {property}
                end: {property + 1}
            specification:
                norm: .inf
                epsilon: {self.epsilon}
        """)
            


if __name__ == "__main__":
    gps = GeneratePropertyScripts()
    gps.main()