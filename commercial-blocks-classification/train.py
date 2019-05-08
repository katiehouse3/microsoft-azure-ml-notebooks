# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Manual train.py for Classification of Commericial Blocks
# ~ By: Katie House
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# IMPORT LIBRARIES
from sklearn.datasets import load_svmlight_file
import numpy as np
import time
from sklearn.metrics import f1_score
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# IMPORT AZURE LIBRARY
from azureml.core.run import Run


# DEFINE FUNCTIONS
def get_data(filepath):
    data = load_svmlight_file(filepath)
    return data[0], data[1]


def Classifier_Test_Train(model, model_name):
    # Training Model...
    start_time = time.time()  # track train time
    model.fit(X_train, y_train)
    train_t = round(time.time() - start_time, 2)

    # Testing Model...
    predictions = model.predict(X_test)
    accuracy = round(f1_score(y_test, predictions), 3)

    # AZURE LOGGING VARIABLES
    run_logger = Run.get_context()
    run_logger.log(name='Model', value=model_name)
    run_logger.log(name='Accuracy', value=accuracy)
    run_logger.log(name='Training_Time', value=train_t)


# MAIN FUNCTION
if __name__ == '__main__':
    run = Run.get_context()

    # IMPORT DATA
    X_train, y_train = get_data("data/train_data.txt")
    X_test, y_test = get_data("data/test_data.txt")

    X_train = X_train.toarray()  # convert sparce matrix to array
    X_test = X_test.toarray()

    # ITERATE THROUGH MODELS
    model_list = [RandomForestClassifier(),
                  MLPClassifier(),
                  KNeighborsClassifier()]
    model_names = ["Random Forest",
                   "Neural Networks",
                   "K Nearest Neighbor"]

    # TRAIN AND TEST MODELS
    results = []
    for i in range(len(model_list)):
        results += [(Classifier_Test_Train(model_list[i], model_names[i]))]
