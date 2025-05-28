import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, NMF
import os


df = pd.read_csv('data_number_nine.csv', header='infer')
df.shape