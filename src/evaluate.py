import pandas as pd
import pickle
import yaml
from sklearn.metrics import accuracy_score
import os
import mlflow
from urllib.parse import urlparse

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Pavankola76/machinelearningpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="Pavankola76"
os.environ["MLFLOW_TRACKING_PASSWORD"]= "0c1b18c5ae8aa12319128c659f3ec21c25928939" # This is the token or secret access key.

# load the parameters from params.yaml, since we use the same data for evaluation as well.
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/Pavankola76/machinelearningpipeline.mlflow")
    #load the model from the disk
    model=pickle.load(open(model_path,'rb'))
    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)
    # log metrics to mlflow
    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")

if __name__ == "__main__":
    evaluate(params["data"],params["model"])