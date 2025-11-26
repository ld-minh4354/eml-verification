import os, sys
import re
import pandas as pd



class RawResult:
    def __init__(self):
        self.add_project_folder_to_pythonpath()
        self.folder = os.path.join("logs_verification")
        self.df = pd.DataFrame(columns=["file_name",
                                        "dataset", "model_type", "seed",
                                        "epsilon", "verifier", "property",
                                        "result"])


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)

    
    def main(self):
        for file_name in os.listdir(self.folder):
            file_path = os.path.join(self.folder, file_name)

            with open(file_path, "r") as f:
                content = f.read()
            
            self.process_file(file_name, content)
        
        os.makedirs("results", exist_ok=True)
        self.df.to_csv(os.path.join("results", "raw_result.csv"), index=False)

        self.group_by_model_and_seed()
        self.df.to_csv(os.path.join("results", "result_by_model_and_seed.csv"), index=False)

        self.group_by_model()
        self.df.to_csv(os.path.join("results", "result_by_model.csv"), index=False)
    

    def process_file(self, file_name, content):
        self.error = False
        dataset = self.regex_helper(file_name, content, "DATASET")
        model_type = self.regex_helper(file_name, content, "MODEL TYPE")
        seed = self.regex_helper(file_name, content, "SEED")
        prop = self.regex_helper(file_name, content, "PROPERTY NO")
        verifier = self.regex_helper(file_name, content, "VERIFIER")
        epsilon = self.regex_helper(file_name, content, "EPSILON")

        if "result: unsat" in content:
            result = 1
        else:
            result = 0

        self.df.loc[len(self.df)] = {
                    "file_name": file_name,
                    "dataset": dataset,
                    "model_type": model_type,
                    "seed": seed,
                    "epsilon": epsilon,
                    "verifier": verifier,
                    "property": prop,
                    "result": result
                }
        

    def group_by_model_and_seed(self):
        self.df = (
            self.df
            .groupby(["dataset", "model_type", "seed", "epsilon", "property"], as_index=False)
            .agg(result=("result", "max"))
        )

        self.df = (
            self.df
            .groupby(["dataset", "model_type", "seed", "epsilon"], as_index=False)
            .agg(result=("result", "sum"))
        )

        self.df = self.df.sort_values(["dataset", "model_type", "seed", "epsilon"])


    def group_by_model(self):
        self.df = (
            self.df
            .groupby(["dataset", "model_type", "epsilon"], as_index=False)
            .agg(result=("result", "sum"))
        )

        self.df = self.df.sort_values(["dataset", "model_type", "epsilon"])


    def regex_helper(self, file_name, content, header):
        pattern = rf"{header}:\s*(\S+)"
        match = re.search(pattern, content)
        if match:
            result = match.group(1)
            return result
        else:
            print(f"Error processing {header} in {file_name}")
            return None



if __name__ == "__main__":
    rr = RawResult()
    rr.main()
