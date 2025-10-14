import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any

# Importa todas as funções de modelagem do arquivo modeling_tools.py
# O arquivo modeling_tools.py deve estar no mesmo diretório ou em um PATH conhecido.
import modeling_tools 

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

TARGET_COL: str = 'Target'
BASE_PATH: str = '../../data/processed/'
MODELS_DIR: str = 'models/'

FILE_PATHS: Dict[str, str] = {
'X_train': os.path.join(BASE_PATH, 'X_train.csv'),
'X_test': os.path.join(BASE_PATH, 'X_test.csv'),
'X_oot': os.path.join(BASE_PATH, 'X_oot.csv'),
'y_train': os.path.join(BASE_PATH, 'y_train.csv'),
'y_test': os.path.join(BASE_PATH, 'y_test.csv'),
'y_oot': os.path.join(BASE_PATH, 'y_oot.csv'),
}

# 1. Grid de Hiperparâmetros para Regressão Logística (Modelo Baseline)
LOG_PARAM_GRID: Dict[str, List[Any]] = {
    'C': [0.01, 0.1, 1, 10], #regularization strength
    'penalty': ['l1', 'l2'] 
}

# 2. Grid de Hiperparâmetros para XGBoost (Modelo Avançado)
XGB_PARAM_GRID: Dict[str, List[Any]] = {
    'max_depth': [3, 5, 7, 10, 20],
    'n_estimators': [100, 200, 250, 300, 350, 400, 500, 700],
    'learning_rate': [0.002, 0.005, 0.01, 0.05, 0.1],
}

SVM_PARAM_GRID: Dict[str, List[Any]] = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
    }

MLP_PARAM_GRID: Dict[str, List[Any]] = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant','adaptive'],
}






# ==============================================================================
# FUNÇÃO PRINCIPAL DE APLICAÇÃO
# ==============================================================================

def run_modeling_pipeline():
    """
    Function to run the entire modeling pipeline choosing between Logistic Regression and XGBoost.
    """
    
    print("--- INICIANDO PIPELINE DE MODELAGEM ---")
    
    # 1. DATA LOADING
    try:
        X_train = pd.read_csv(FILE_PATHS['X_train'])
        X_test = pd.read_csv(FILE_PATHS['X_test'])
        X_oot = pd.read_csv(FILE_PATHS['X_oot'])
        
        # Targets (Y) - Solução para o erro de 'inconsistent samples'
    
        y_train_df = pd.read_csv(FILE_PATHS['y_train'], header=None)
        y_test_df = pd.read_csv(FILE_PATHS['y_test'], header=None)
        y_oot_df = pd.read_csv(FILE_PATHS['y_oot'], header=None)

        # PASSO CRÍTICO: Remover a primeira linha do Y lido se o X tinha cabeçalho.
        # O [1:] garante que o Y tenha o mesmo número de linhas de dados que o X.
        y_train = y_train_df.iloc[1:, 0].rename(TARGET_COL).astype(int).reset_index(drop=True)
        y_test = y_test_df.iloc[1:, 0].rename(TARGET_COL).astype(int).reset_index(drop=True)
        y_oot = y_oot_df.iloc[1:, 0].rename(TARGET_COL).astype(int).reset_index(drop=True)

        # PASSO EXTRA: Garantir que o índice de Y comece em 0, alinhado ao X.
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        X_oot = X_oot.reset_index(drop=True)
        
        len(y_train), len(y_test), len(y_oot)
        print(f"Dados carregados com sucesso de '{BASE_PATH}'.")
    except FileNotFoundError:
        print(f"\nERRO: Arquivos de dados não encontrados.")
        print(f"Por favor, verifique se a pasta '{BASE_PATH}' existe e contém os arquivos.")
        return

    
    # 3. MODELING DIRECTORY CHECK
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Diretório de modelos criado: '{MODELS_DIR}'.")


    # 4. STANDARDIZATION - Essential for Logistic Regression
    if len(X_test) > 0:
        X_train_scaled, X_test_scaled = modeling_tools.standardize_features(X_train, X_test)
    else:
        X_train_scaled, X_test_scaled = modeling_tools.standardize_features(X_train, X_oot)
        # if test is all out of time
    
    # Salva o Scaler para ser usado na inferência de novos dados
    
    modeling_tools.fit_and_save_scaler(X_train, os.path.join(MODELS_DIR, 'feature_scaler.joblib'))

    # --- MODELO 1: REGRESSÃO LOGÍSTICA (BASELINE) ---
    print("\n" + "="*50)
    best_log_model = modeling_tools.train_model_with_gridsearch(
        model_name='LogisticRegression',
        X_train=X_train_scaled,
        y_train=y_train,
        param_grid=LOG_PARAM_GRID,
        target_col=TARGET_COL,
        cv_folds=3
    )
    modeling_tools.save_model(best_log_model, os.path.join(MODELS_DIR, 'logistic_best.joblib'))


    # --- MODELO 2: XGBOOST ---
    print("\n" + "="*50)
    
    best_xgb_model = modeling_tools.train_model_with_gridsearch(
        model_name='XGBoost',
        X_train=X_train_scaled,
        y_train=y_train,
        param_grid=XGB_PARAM_GRID,
        target_col=TARGET_COL,
        cv_folds=3
    )
    modeling_tools.save_model(best_xgb_model, os.path.join(MODELS_DIR, 'xgboost_best.joblib'))

    # --- MODELO 3: MLP ---
    print("\n" + "="*50)
    
    best_mlp_model = modeling_tools.train_model_with_gridsearch(
        model_name='MLPClassifier',
        X_train=X_train_scaled,
        y_train=y_train,
        param_grid=MLP_PARAM_GRID,
        target_col=TARGET_COL,
        cv_folds=3
    )
    modeling_tools.save_model(best_mlp_model, os.path.join(MODELS_DIR, 'mlp_best.joblib'))

    # --- MODELO 4: SVM ---
    print("\n" + "="*50)
    best_svm_model = modeling_tools.train_model_with_gridsearch(
        model_name='SVM',
        X_train=X_train_scaled,
        y_train=y_train,
        param_grid=SVM_PARAM_GRID,
        target_col=TARGET_COL,
        cv_folds=3
    )
    modeling_tools.save_model(best_svm_model, os.path.join(MODELS_DIR, 'svm_best.joblib'))

    

    print("\n--- MODELAGEM CONCLUÍDA ---")

if __name__ == '__main__':
    run_modeling_pipeline()