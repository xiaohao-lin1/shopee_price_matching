import numpy as np, pandas as pd, gc
import matplotlib.pyplot as plt
import tensorflow as tf
import os
# import cv2, matplotlib.pyplot as plt
# import cudf, cuml, cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.neighbors import NearestNeighbors

from tensorflow.keras.applications import EfficientNetB0
# print('RAPIDS',cuml.__version__)
print('TF',tf.__version__)

#training data location
wd = 'E:\shopee-product-matching'
os.chdir(wd)
train_add = 'train.csv'
test_add = 'test.csv'

train = pd.read_csv(train_add)
train.head()