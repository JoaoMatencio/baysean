import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pystan

df = pd.read_csv('german_credit_data_preprocessed.csv')
df.columns