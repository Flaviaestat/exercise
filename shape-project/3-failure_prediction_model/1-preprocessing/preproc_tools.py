import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder     

# ==============================================================================
# PREPROCESSING TOOLS  FOR FAILURE PREDICTION MODEL
# ==============================================================================


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a DataFrame showing the percentage and count of missing values per column.

    Args:
        df: The input DataFrame to analyze.

    Returns:
        Prints total missing values in the DataFrame.
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    print("Total missing values in DataFrame:", missing_count.sum())


def report_outliers(df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
    """
    Generates a DataFrame reporting the number of outliers (based on the IQR method)
    for specified numerical columns.

    Args:
        df: The input DataFrame to analyze.
        numerical_cols: List of numerical columns to check.

    Returns:
        DataFrame with 'Outlier Count' and 'Outlier Percent' columns.
    """
    outlier_data: Dict[str, Tuple[int, float]] = {}
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outlier_percent = (outlier_count / len(df)) * 100
        
        print(f"Column: {col}, Outlier Count: {outlier_count}, Outlier Percent: {outlier_percent:.2f}%")

def create_rolling_features(df: pd.DataFrame, window_map: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Cria features baseadas em janelas móveis (rolling features), como Média e Desvio Padrão.

    A função deve ser aplicada APENAS nas colunas de sinal contínuo (Temperatura, Pressão, etc.).

    Args:
        df: DataFrame de entrada, OBRIGATORIAMENTE ordenado por tempo/ciclo.
        window_map: Dicionário mapeando o nome da coluna para uma lista de tamanhos de janelas (windows).
                    Ex: {'Temperature': [5, 10], 'Vibrations_Y': [3, 7]}.

    Returns:
        DataFrame com colunas de rolling features adicionadas.
    """
    df_rolling = df.copy()
    
    print("\nIniciando Engenharia de Features (Rolling Windows):")

    # Garante que o DataFrame seja tratado como série temporal para o rolling
    
    for col, windows in window_map.items():
        if col not in df_rolling.columns:
            print(f"  -> Aviso: Coluna '{col}' não encontrada para Rolling Feature.")
            continue
            
        for window in windows:
            # 1. Cálculo da Média Móvel (Rolling Mean)
            new_mean_col = f'{col}_RollMean_{window}'
            # .shift(1) é crucial! Garante que a média termine no ciclo anterior (t-1),
            # evitando Data Leakage (usar a informação do próprio ciclo atual).
            df_rolling[new_mean_col] = df_rolling[col].rolling(window=window).mean().shift(1)
            
            # 2. Cálculo do Desvio Padrão Móvel (Rolling Standard Deviation)
            new_std_col = f'{col}_RollStd_{window}'
            df_rolling[new_std_col] = df_rolling[col].rolling(window=window).std().shift(1)
            
            print(f"  -> Features de Rolling (Mean/Std, Janela={window}) criadas para '{col}'.")
    
    # Retorna o DataFrame para que o próximo passo possa ser aplicado
    return df_rolling.reset_index(drop=True)


def create_derivative_features(df: pd.DataFrame, window_map: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Cria features de derivada (ou taxa de mudança/slope) sobre uma janela móvel.
    Calcula (Valor_t - Valor_{t-N}) / N.

    A função deve ser aplicada APENAS nas colunas de sinal contínuo (Temperatura, Pressão, etc.).

    Args:
        df: DataFrame de entrada, OBRIGATORIAMENTE ordenado por tempo/ciclo.
        window_map: Dicionário mapeando a coluna para uma lista de tamanhos de janelas (windows)
                    sobre as quais a derivada será calculada.
                    Ex: {'Pressure': [5, 10], 'Vibrations_Y': [3]}.

    Returns:
        DataFrame com colunas de derivada adicionadas.
    """
    df_derivative = df.copy()
    
    print("\nIniciando Engenharia de Features (Derivadas/Slope):")

    for col, windows in window_map.items():
        if col not in df_derivative.columns:
            print(f"  -> Aviso: Coluna '{col}' não encontrada para Derivada.")
            continue
            
        for window in windows:
            # 1. Calcular a diferença entre o valor atual (t) e o valor N ciclos atrás (t-N).
            # O .diff(window) do Pandas faz exatamente (Valor_t - Valor_{t-N}).
            diff = df_derivative[col].diff(periods=window)
            
            # 2. Normalizar pelo tamanho da janela (Slope = Delta Y / Delta X)
            # Como Delta X é o tamanho da janela (window), dividimos pela janela.
            new_slope_col = f'{col}_Slope_{window}'
            df_derivative[new_slope_col] = diff / window
            
            # 3. CRUCIAL: Aplicar shift(1) para evitar Data Leakage
            # Queremos que o Slope (t) use o sinal ATÉ o ciclo t-1 para prever o ciclo t.
            df_derivative[new_slope_col] = df_derivative[new_slope_col].shift(1)
            
            print(f"  -> Feature de Derivada/Slope (Janela={window}) criada para '{col}'.")
            
    return df_derivative

def create_lagged_features(df: pd.DataFrame, lag_map: Dict[str, int]) -> pd.DataFrame:
    """
    Creates lagged features for numerical variables and the target.

    Args:
        df: Input DataFrame, MUST be sorted by time/cycle.
        lag_map: Dictionary mapping column name to the maximum lag number.
                 E.g.: {'Temperature': 3, 'Fail': 10}.

    Returns:
        DataFrame with lagged columns added.
    """
    df_lagged = df.copy()
    
    print("\nStarting Feature Engineering (Lags):")

    for col, max_lag in lag_map.items():
        if col not in df_lagged.columns:
            print(f"  -> Warning: Column {col} not found in DataFrame.")
            continue
            
        for lag in range(1, max_lag + 1):
            new_col_name = f'{col}_Lag{lag}'
            # shift() moves values down, creating the lag.
            # The initial value (lag=1) will be NaN.
            df_lagged[new_col_name] = df_lagged[col].shift(lag)
        
        print(f"  -> {max_lag} Lags created for column '{col}'.")
            
    return df_lagged



def handle_missing_values(df: pd.DataFrame, numerical_cols: List[str], strategy: str = 'median') -> pd.DataFrame:
    """
    Handles missing values in numerical columns using the specified imputation strategy.

    Args:
        df: Input DataFrame.
        numerical_cols: List of numerical columns.
        strategy: 'median' or 'mean' for imputation.

    Returns:
        DataFrame with missing values filled.
    """
    df_treated = df.copy()
    
    for col in numerical_cols:
        if df_treated[col].isnull().any():
            if strategy == 'median':
                imputation_value = df_treated[col].median()
            elif strategy == 'mean':
                imputation_value = df_treated[col].mean()
            else:
                raise ValueError("Invalid strategy. Use 'median' or 'mean'.")
                
            df_treated[col] = df_treated[col].fillna(imputation_value)
            print(f"  -> Missing values in {col} handled with {strategy} ({imputation_value:.2f}).")
            
    return df_treated

def adjust_categorical_variables(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Adjusts categorical variables using One-Hot Encoding (dummy variables).
    Ideal for Presets (Preset_1 and Preset_2).

    Args:
        df: Input DataFrame.
        categorical_cols: List of categorical columns to be encoded.

    Returns:
        DataFrame with categorical columns replaced by dummy variables.
    """
    df_encoded = df.copy()
    
    # Use pd.get_dummies for One-Hot Encoding
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True, prefix=categorical_cols)
    print(f"  -> Categorical variables {categorical_cols} encoded.")
    
    return df_encoded



def split_time_series_data(
    df: pd.DataFrame, 
    target_col: str,
    oot_size: float = 0.1, 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits the DataFrame into Train, Test, and Out-of-Time (OOT) sets 
    sequentially (Time Series split).

    Args:
        df: Input DataFrame (pre-processed and with lags).
        target_col: Name of the target column ('Fail').
        oot_size: Proportion for the Out-of-Time set (e.g., 0.1 for 10%).
        test_size: Proportion for the Test set (e.g., 0.2 for 20%).

    Returns:
        A tuple containing (X_train, X_test, X_oot, y_train, y_test, y_oot).
    """
    
    # 1. Drop rows with NaN (Created by Lags)
    # Crucial to remove initial rows where lags are NaN to prevent issues during training.
    df_clean = df.dropna().reset_index(drop=True)
    
    # 2. Define sequential sizes
    n_samples = len(df_clean)
    n_oot = int(n_samples * oot_size)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test - n_oot
    
    if n_train <= 0:
        raise ValueError("Insufficient training size after dropping NaNs. Adjust sizes or lag.")

    # 3. Sequential Split (Train -> Test -> OOT)
    
    # OOT (Last n_oot cycles)
    df_oot = df_clean.iloc[n_samples - n_oot:]
    
    # Test (Before OOT)
    df_test = df_clean.iloc[n_train : n_train + n_test]
    
    # Train (First n_train cycles)
    df_train = df_clean.iloc[:n_train]
    
    # 4. Separate Features (X) and Target (y)
    
    features = [col for col in df_clean.columns if col != target_col]
    
    X_train, y_train = df_train[features], df_train[target_col]
    X_test, y_test = df_test[features], df_test[target_col]
    X_oot, y_oot = df_oot[features], df_oot[target_col]

    print("\nDivisão Sequencial (Time Series Split) Concluída:")
    print(f"  -> Treino: {len(X_train)} amostras ({len(X_train)/n_samples:.1%})")
    print(f"  -> Teste: {len(X_test)} amostras ({len(X_test)/n_samples:.1%})")
    print(f"  -> OOT: {len(X_oot)} amostras ({len(X_oot)/n_samples:.1%})")
    
    return X_train, X_test, X_oot, y_train, y_test, y_oot
