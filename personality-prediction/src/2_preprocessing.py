import pandas as pd
import numpy as np
import os


# entrando na pasta src/data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocessing_utils import PreprocessData as PreprocessData


os.chdir('../src/data')

# carregando o dataset
df = pd.read_csv('combined_data.csv')
print(len(df))


