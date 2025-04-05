from sklearn.utils import shuffle
import pandas as pd
import numpy as np

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return shuffle(data, random_state=42)

# unique_values - data[column].unique()
# split_dataset - data[data[column] == value], data[data[column] != value]
# most_common_label - data[target].mode()[0]
# entropy - calculeaza proporția, urmată de -sum(p*log2(p))#

def unique_values(data, column):
    return data[column].unique()

def split_dataset(data, column, value):
    return data[data[column] == value], data[data[column] != value]

def most_common_label(data, target):
    return data[target].mode()[0]

def entropy(data, target):
    total = len(data)
    proportions = data[target].value_counts() / total
    return -sum(proportions * np.log2(proportions))