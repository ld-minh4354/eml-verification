import os, sys
import textwrap

import pandas as pd



class GeneratePropertyScripts:
    def __init__(self, dataset = "CIFAR10"):
        self.add_project_folder_to_pythonpath()
        self.dataset = dataset


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def main(self):
        self.get_labels()
        self.generate_scripts()


    def get_labels(self):
        self.df_labels = pd.read_csv(os.path.join("test_data", self.dataset, "labels.csv"))


    def generate_scripts(self):
        os.makedirs(os.path.join("properties", self.dataset), exist_ok=True)

        for _, row in self.df_labels.iterrows():
            index = row["index"]
            label = row["label"]

            file_path = os.path.join("properties", self.dataset, f"property_{index}.py")
            file_content = self.get_file_content(index, label)

            with open(file_path, "w") as f:
                f.write(file_content)


    def get_file_content(self, index, label):
        return textwrap.dedent(f"""\
            import os
            from dnnv.properties import *
            import numpy as np

            N = Network("N")
            x = Image(os.path.join("test_data", "{self.dataset}", "inputs", "input_{index}.npy"))

            epsilon = Parameter("epsilon", type=float)
            true_class = {label}

            Forall(
                x_,  
                Implies(
                    ((x - epsilon) < x_ < (x + epsilon)) & (0 < x_ < 1),
                    np.argmax(N(x_)) == true_class,
                ),
            )
        """)
            


if __name__ == "__main__":
    gps = GeneratePropertyScripts(dataset="CIFAR10")
    gps.main()