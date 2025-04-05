import os
import pandas as pd

from src.utils import load_dataset

dataset_path = os.path.join("data", "diabetes_dataset.csv")
data = load_dataset(dataset_path)

target = "diabet"
train_size = int(0.8*len(data))
train_data = data[:train_size]
test_data = data[train_size:]

