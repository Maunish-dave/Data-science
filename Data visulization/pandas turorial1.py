import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")
help(data)
data.count()
df1 = pd.DataFrame([[1,np.nan]])
df2 = pd.DataFrame([[3,4]])
df1.combine_first(df2)
data.corr()

