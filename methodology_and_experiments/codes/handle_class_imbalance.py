# TMU MRP 2025
# Handle Class Imbalance Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
from collections import Counter

y_train = pd.read_csv("y_train.csv").squeeze()

class_counts = Counter(y_train)
print("Class distribution:", class_counts)

scale_weight = class_counts[0] / class_counts[1]
print("scale_pos_weight:", round(scale_weight, 2))