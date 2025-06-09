import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os
import mlflow
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse # for getting the scheme for mlflow 

# Now since we are tracking mlflow logs in Dagshub we need the remote repository information related to logging experiments.
# click on remote and experiments in dagshub
# Here we have two options, we can perform in any way.
# lets perform in a third posible ways which is not mentioned here to connect mlflow and dagshub

# remember the .env file where we give store the API keyes, we are going to do the same here, the name of the vraible should be same (MLFLOW_TRACKING_URI), so instead of this we can create an environmental file and import it.
# according to the documentation we should also provide username and password as well.

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Pavankola76/machinelearningpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="Pavankola76"
os.environ["MLFLOW_TRACKING_PASSWORD"]= "0c1b18c5ae8aa12319128c659f3ec21c25928939" # This is the token or secret access key.

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2) # higher the verbose higher the messages, so to display all results giving verbose is 2
    # if n_jobs = -1 says to use all cores in the computer
    grid_search.fit(X_train,y_train)
    return grid_search
# this functon will return the best hyperparameters after performing the fit and finding the best.

# load the parameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path)
    # we used the raw path instead of preprocessed because we have taken the column names while preprocessing, so dont do that, if done revert back the columns
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]
    # while performing trainign we should also push all of this by the help of mflow to the dagshub repository.
    mlflow.set_tracking_uri("https://dagshub.com/Pavankola76/machinelearningpipeline.mlflow") # instead of this we can also paste the URI present above

    # Start the mlflow run
    with mlflow.start_run():
        # split the dataset into training and test sets
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) # did not take random state because we have assigned it already in the train function
        signature = infer_signature(X_train,y_train) # to get the scheme of the dataset

        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [5,10,None],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2]
        }
        
        # perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)
        
        # get the best model
        best_model = grid_search.best_estimator_

        # predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")

        # now we need to track all of this the accuracy score, best model etc..
        # Log additional metrics \ (should be in the block mlflow start_run)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best_n_estimator",grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth",grid_search.best_params_['max_depth'])
        mlflow.log_param("best_sample_split",grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_sample_leaf",grid_search.best_params_['min_samples_leaf'])
        
        # log the confusion matrix and classification report
        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)
        
        mlflow.log_text(str(cm),"confusion_matrix.txt")  # refer documentation, we have converted cm to string 
        mlflow.log_text(cr,"classification_report.txt")  # the second parameter is a file that contains this information, not in a variable.

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store!='file': # that is if it is a http or https etc
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best model")
        else:
            mlflow.sklearn.log_model(best_model,"model",signature=signature)

        # create the directory to save the model
        os.makedirs(os.path.dirname(model_path),exist_ok=True) # the pickle file mentioned in yaml file
        # now our path is created, we should dump this best model into this created path in the form of pickle file
        filename=model_path
        # for dumping in the pickle file format we first open this folder in wright bite mode
        pickle.dump(best_model,open(filename,'wb'))

        print(f"Model saved to {model_path}")

 # run this
if __name__ == "__main__":
   train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])       # yaml file params

