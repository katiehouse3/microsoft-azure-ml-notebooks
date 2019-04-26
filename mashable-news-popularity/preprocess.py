import os
project_folder = os.getcwd()
print(project_folder)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

## Read data from a website
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

def get_data():
    # Read Data from url
    print('Reading data...')
    resp = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    zipfile.namelist()
    file = 'OnlineNewsPopularity/OnlineNewsPopularity.csv'
    df = pd.read_csv(zipfile.open(file))
    
    # Preprocessing
    # Remove beginning white space in the columns
    print('Stripping off white space...')
    df.rename(columns=lambda x: x.strip(), inplace=True)
    
    # Set Target Label
    # Define number of popularity categories to predict
    print('Make target categories')
    share_categories = [1,2,3,4,5]
    df['share_cat'] = np.array(pd.qcut(df['shares'], 5, share_categories))
    df['share_cat'].dtype
    df['share_cat'] = np.array(df['share_cat'].astype('category'))
    
    # Split Data
    # time delta and url are not predictive attributes, exclude them
    x_df = df[df.columns[2:-2]] # url and time delta are the first two attributes 
    y_df = df[df.columns[-1]]
    
    print('Splitting data...')
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, 
                                                        random_state=607)
    
    return { "X": x_train.values, "y": y_train.values, 
            "X_valid": x_test.values, "y_valid": y_test.values}
