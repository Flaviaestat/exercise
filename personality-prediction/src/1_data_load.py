import pandas as pd
import numpy as np
import os


# entrando na pasta src/data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../src/data') 


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train['source'] = 'train'
df_test['source'] = 'test'

df = pd.concat([df_train, df_test], ignore_index=True)

df.to_csv('combined_data.csv', index=False)
print("Combined data shape:", df.shape)
print("Columns in the combined data:", df.columns.tolist())
