from src.utils import entropy
from src.utils import split_dataset
from src.utils import unique_values
from src.utils import most_common_label
import numpy as np

class Node:
    def __init__(self, column=None, value=None, true_branch=None, false_branch=None, label=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.label = label
    def is_leaf(self):
        return self.label is not None

def best_split(data, target):

    best_gain = 0
    best_col, best_value = None, None
    current_entropy = entropy(data, target)

    for col in data.drop(col=[target]):
        unique_vals = unique_values(data, col)

        for value in unique_vals:
            left_data, right_data = split_dataset(data, col, value)

            if len(left_data) == 0 or len(right_data) == 0:
                continue

            left_entropy = entropy(left_data, target)
            right_entropy = entropy(right_data, target)

            left_weight = len(left_data) / len(data)
            right_weight = len(right_data) / len(data)
            weighted_entropy = left_entropy*left_weight + right_entropy*right_weight

            info_gain = current_entropy - weighted_entropy

            if info_gain > best_gain:
                best_gain = info_gain
                best_col = col
                best_value = value

    return best_col, best_value, best_gain

def build_tree(data, target, max_depth=None, depth=0):
    