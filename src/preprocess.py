import pandas as pd
import sys
import yaml
import os

# loading parameters from the yaml file, i was saying load and open the yaml file and then move the keyvalues of preprocess to params


params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path,output_path):
    data = pd.read_csv(input_path)
    # in the os i am making a directory (mkdir) and giving the directory name (the directory is os.path.dirname)
    # if my path already exist then i am setting to retur true
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.to_csv(output_path,header=None,index=False) # after performing preprocessing covert to .csv 
    print(f"preprocessed data saved to {output_path} ")

if __name__ == "__main__":
    preprocess(params["input"],params["output"])


# there is no preprocessing required here so i have just written the basic structure of preprocessing
# Now lets also execute this code: python src/preprocess.py
# Now we can clearly see that under data another folder named preprocess is created where we can find preprocessed data.
