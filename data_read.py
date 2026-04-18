import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("CALCOFI_DIC_20250122.csv")
df = df.dropna()
df = df.drop(columns=[''])
print(df)