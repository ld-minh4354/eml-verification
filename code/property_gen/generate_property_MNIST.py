import os, sys
import textwrap
import argparse



class GeneratePropertyMNIST:
    def __init__(self, epsilon, job_index):
        self.add_project_folder_to_pythonpath()
        self.epsilon = str(epsilon)
        self.job_index = job_index

        self.model_types = ["FC", "conv"]
        self.seed_values = list(range(0, 250, 5))
        self.property_values = list(range(100))
        
        os.makedirs(os.path.join("properties"), exist_ok=True)


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def generate(self, index):
        num_seeds = len(self.seed_values)
        num_props = len(self.property_values)

        model_index = index // (num_seeds * num_props)
        seed_index = (index % (num_seeds * num_props)) // num_props
        property_index = index % num_props

        model = self.model_types[model_index]
        seed = self.seed_values[seed_index]
        property = self.property_values[property_index]

        self.print_info(model, seed, property)
        file_content = self.get_file_content(model, seed, property)

        file_path = os.path.join("properties", f"MNIST_{self.epsilon}_{self.job_index}.yaml")
        with open(file_path, "w") as f:
            f.write(file_content)


    def print_info(self, model, seed, property):
        print(f"DATASET: MNIST")
        print(f"MODEL TYPE: {model}")
        print(f"SEED: {seed}")
        print(f"PROPERTY: {property}")
        print(f"VERIFIER: abc")
        print(f"EPSILON: {self.epsilon}")


    def get_file_content(self, model, seed, property):
        if model == "FC":
            model_name = "mnist_fc_3_512"
        elif model == "conv":
            model_name = "mnist_conv_big"
        return textwrap.dedent(f"""\
            model:
                name: {model_name}
                path: models/MNIST/{model}/MNIST_{model}_{seed}.pth
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--index", type=int)
    parser.add_argument("--job", type=int)
    args = parser.parse_args()

    gps = GeneratePropertyMNIST(epsilon=args.epsilon, job_index=args.job)
    gps.generate(index=args.index)