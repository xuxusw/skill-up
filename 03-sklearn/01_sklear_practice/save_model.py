from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os
from sklearn.externals import joblib

X = []
y = []
scaler = StandardScaler()
dir_counter = 0

for i in range(9):
    files = os.listdir(f'./data/{i}')