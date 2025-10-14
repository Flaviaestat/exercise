
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import preproc_tools as pp

## GETTING DATA
df = pd.read_csv('../../data/Test O_G_Equipment_Data.csv', sep=';', decimal=',')

df = df.sort_values(by=['Cycle']).reset_index(drop=True)

print(f'{len(df)} rows and {len(df.columns)} columns loaded.')

## DEFININIG INITIAL FEATURES

# CODING TARGET VARIABLE
prediction_cenario = "PREDICT_AHEAD"  # Options: "FAILURE_PREDICTION", "FIRST_FAILURE", "PREDICT_AHEAD"
LOOK_AHEAD_WINDOW = 5  # Used only if prediction_cenario is "PREDICT_AHEAD"
print(f"Prediction Scenario: {prediction_cenario}")

if prediction_cenario == "FAILURE_PREDICTION":
    df['Target'] = np.where(df['Fail'] == "VERDADEIRO", 1, 0)
elif prediction_cenario == "FIRST_FAILURE":
    df['Target'] = np.where(df['Fail'] == "VERDADEIRO", 1, 0)
    df['previous_fail'] = df['Target'].shift(1, fill_value=0)
    df['Target'] = (df['Target'] == 1) & (df['previous_fail'] == 0)
    del df['previous_fail']
elif prediction_cenario == "PREDICT_AHEAD":
    df['Target'] = np.where(df['Fail'] == "VERDADEIRO", 1, 0)
    df['Start_of_Failure'] = ((df['Target'].astype(int) == 1) & (df['Target'].astype(int).shift(1, fill_value=0) == 0)).astype(int)
    df['Target_LookAhead'] = df['Start_of_Failure'].rolling(
        window=LOOK_AHEAD_WINDOW + 1,
        min_periods=1
    ).max().shift(-LOOK_AHEAD_WINDOW, fill_value=0)
    
    df['Target'] = df['Target_LookAhead'].fillna(0).astype(int) # Preenche NaNs no final da série com 0
    del df['Start_of_Failure']
    del df['Target_LookAhead']


print(df['Target'].value_counts())

df['Target'] = df['Target'].astype(int)


print('Removing non-predictive columns...')
del df['Fail']
del df['Cycle']

## MAIN PIPELINE FOR PREPROCESSING

print('Selecting feature types...')
NUMERIC_COLS = df.select_dtypes(include=['float64']).columns
CATEGORICAL_COLS = df.select_dtypes(include=['int']).drop('Target', axis = 1).columns
TARGET = 'Target'
print(f'Numeric Columns: {NUMERIC_COLS}')
print(f'Categorical Columns: {CATEGORICAL_COLS}')
print(f'Target Column: {TARGET}')
    
# Lag Mapping (Based on EDA)
LAG_MAPPING: Dict[str, int] = {
    'Temperature': 3,
    'Pressure': 5,
    'VibrationX': 8,
    'VibrationY': 8,
    'VibrationZ': 8,
    'Frequency': 8,
    'Target': 10 # Target Lag
}

ROLLING_MAP: Dict[str, List[int]] = {
    'Temperature': [10, 30],  # Tendência de aquecimento
    'Pressure': [3, 7],       # Aumento rápido de pressão
    'Vibrations_Y': [5, 15],  # Instabilidade e aumento sustentado
    'Frequency': [5]
}

DERIVATIVE_MAP: Dict[str, List[int]] = {
    'Pressure': [3, 5],      # Velocidade de aumento de pressão (curto prazo)
    'Vibrations_Y': [5, 10], # Velocidade de aumento de instabilidade
    'Temperature': [10]      # Velocidade de aquecimento
}

# Creating flag high value all variables - important for logistic regression
df['flag_high_values'] = np.where((df.Temperature > 90) 
                                  & (df.Pressure > 90)
                                  & (df.VibrationX > 90) 
                                  & (df.VibrationY > 90)
                                & (df.VibrationZ > 90), 1, 0)

print(df['flag_high_values'].value_counts())

print("----- PIPELINE EXECUTION -----")
print("\nRelatório de Missings:\n", pp.report_missing_values(df.copy()))
print("\nRelatório de Outliers (IQR):\n", pp.report_outliers(df.copy(), NUMERIC_COLS))


# 1. Handle Missings (Using Median)
df_processed = pp.handle_missing_values(df.copy(), NUMERIC_COLS, strategy='median')

# 2. Handle Outliers (Capping)
#df_processed = pp.cap_outliers_iqr(df_processed, NUMERIC_COLS)

# 3. Feature Engineering (Lags and rolling windows)
# The DataFrame MUST be sorted here!
df_processed = pp.create_rolling_features(df_processed, 
                                          window_map=ROLLING_MAP)

df_processed = pp.create_derivative_features(df_processed, DERIVATIVE_MAP)

df_lagged = pp.create_lagged_features(df_processed, LAG_MAPPING)

# 4. Adjust Categorical Variables (One-Hot Encoding)
df_final = pp.adjust_categorical_variables(df_lagged, CATEGORICAL_COLS)


print(f"\nDataFrame Final Shape: {df_final.shape}")
print(df_final.head(5))

print(df_final.dtypes)

df_final.to_csv('../../data/processed/df_final.csv', index=False)
print("\nDataFrame final saved to '../../data/processed/df_final.csv'.")

print("\n----- PASSO 3: DIVISÃO TREINO, TESTE, OOT -----")

# 5. Train, Test, and OOT Split
X_train, X_test, X_oot, y_train, y_test, y_oot = pp.split_time_series_data(
    df_final, 
    target_col=TARGET,
    oot_size=0.2,  # 10% for Out-of-Time
    test_size=0.0  # 20% for Test
)

# Example of class balance check
print("\nClass Balance (Train Target):\n", y_train.value_counts(normalize=True))

# Creating directory if not exists
import os
os.makedirs('../../data/processed/', exist_ok=True) 

X_train.to_csv('../../data/processed/X_train.csv', index=False)
X_test.to_csv('../../data/processed/X_test.csv', index=False)
X_oot.to_csv('../../data/processed/X_oot.csv', index=False)
y_train.to_csv('../../data/processed/y_train.csv', index=False)
y_test.to_csv('../../data/processed/y_test.csv', index=False)
y_oot.to_csv('../../data/processed/y_oot.csv', index=False)

print("\nPreprocessing and data splitting completed. Processed files saved to '../../data/processed/' directory.")


