print("\nClassification of Commercial Blocks")
print("By: Katie House")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORT LIBRARIES
from sklearn.datasets import load_svmlight_file
import numpy as np
import time
from sklearn.metrics import f1_score
import csv
from sklearn.preprocessing import RobustScaler   
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from azureml.core.run import Run

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DEFINE FUNTIONS
# Data Upload Functions
def get_data(filepath):
    data = load_svmlight_file(filepath)
    return data[0], data[1]

# Default Classification Function
def Classifier_Test_Train(model, model_name):
	print("\n%s Classification" % model_name)
	print("Training Model...")
	start_time = time.time() # track train time
	model.fit(X_train, y_train)
	train_t = round(time.time() - start_time,2)

	print("Testing Model...")
	start_time = time.time() # track train time
	predictions = model.predict(X_test)
	accuracy = round(f1_score(y_test, predictions),3)
	predict_t = round(time.time() - start_time,2)
	
	table_row = [model_name, accuracy, train_t, predict_t]
	run_logger = Run.get_context()
	run_logger.log(name='Model', value=model_name)
	run_logger.log(name='Accuracy', value=accuracy)
	run_logger.log(name='Training_Time', value=train_t)
	return table_row

run = Run.get_context() 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORT DATA
print("\nImporting Data...")

X_train, y_train = get_data("data/train_data.txt")
X_test, y_test = get_data("data/test_data.txt")

X_train = X_train.toarray() # convert sparce matrix to array
X_test = X_test.toarray() 
print("Data imported.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROBLEM 3a ~ DEFAULT CLASSIFIER ACCURACY TABLE
print('~' * 60)
print("PROBLEM 3: DEFAULT CLASSIFIER ACCURACY")

# Initialize parameters
model_list = [RandomForestClassifier(), \
					MLPClassifier(), \
					KNeighborsClassifier()]
model_names = ["Random Forest",\
			 	"Neural Networks",\
			 	"K Nearest Neighbor"]
results = []

# Train and Test Models
for i in range(len(model_list)):
	results += [(Classifier_Test_Train(model_list[i], model_names[i]))]

# CREATE default accuracy TABLE
print("\nClassifier Output Summary:\n")
for i in range(len(results)):
	cols = ["Classifier", "Test Accuracy (F1)", "Train Time (s)", "Predict Time (s)"]
	title = '|'.join(str(x).ljust(20) for x in cols)
	rows = '|'.join(str(x).ljust(20) for x in results[i])
	if i == 0:
		print('-' * len(title))
		print(title)
		print('-' * len(title))
	print(rows)
print('-' * len(title))