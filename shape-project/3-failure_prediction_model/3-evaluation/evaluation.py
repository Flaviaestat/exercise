#%%

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any
import evaluation_tools
#import modeling_tools # Para carregar o scaler


# ==============================================================================
# CONFIGURATIONS
# ==============================================================================

TARGET_COL: str = 'Target'
BASE_PATH: str = '../../data/processed/'
MODELS_DIR: str = '../training/models/'
PLOTS_DIR: str = 'plots/'
REPORTS_DIR: str = 'reports/'

# Models artefact paths
MODEL_PATHS: Dict[str, str] = {
    'LogisticRegression': os.path.join(MODELS_DIR, 'logistic_best.joblib'),
    'XGBoost': os.path.join(MODELS_DIR, 'xgboost_best.joblib'),
    'SVM': os.path.join(MODELS_DIR, 'svm_best.joblib'),
    'MLPClassifier': os.path.join(MODELS_DIR, 'mlp_best.joblib')
}


# Scalr artefact path
SCALER_PATH: str = os.path.join(MODELS_DIR, 'feature_scaler.joblib')


FILE_PATHS: Dict[str, str] = {
    'X_test': os.path.join(BASE_PATH, 'X_test.csv'),
    #'X_train': os.path.join(BASE_PATH, 'X_train.csv'),
    'X_oot': os.path.join(BASE_PATH, 'X_oot.csv'),
    'y_test': os.path.join(BASE_PATH, 'y_test.csv'),
    'y_oot': os.path.join(BASE_PATH, 'y_oot.csv'),
}

#COMENTAR DEPOIS
'''
X_test = pd.read_csv(FILE_PATHS['X_test'])
X_oot = pd.read_csv(FILE_PATHS['X_oot'])
        
        # Carrega Targets (Y) - Lógica corrigida para desalinhamento e tipo
y_test_df = pd.read_csv(FILE_PATHS['y_test'], header=None)
y_oot_df = pd.read_csv(FILE_PATHS['y_oot'], header=None)

#Carreg Train pra comparar
X_train = pd.read_csv(FILE_PATHS['X_train'])

print('X TRAIN')
print(X_train.dtypes)
print('X TEST')
print(X_test.dtypes)
print('OOT')
print(X_oot.dtypes)
print('Y TEST')
print(y_test_df.dtypes)
print('Y OOT')
print(y_oot_df.dtypes)

'''


#CLASSIFICATION_THRESHOLD: float = 0.5


def load_data_and_scaler() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Any]:
    """
    Carrega dados X e y de Teste e OOT e o StandardScaler.
    Aplica conversão de tipo seletiva e alinhamento de índices para resolver 
    o erro 'The feature names should match'.
    """
    
    print("--- 1. CARREGANDO DADOS E SCALER ---")
    
    try:
        # Carrega Features (X). Preserva o header.
        X_test = pd.read_csv(FILE_PATHS['X_test'])
        X_oot = pd.read_csv(FILE_PATHS['X_oot'])
        
        # Carrega Targets (Y) - Lógica corrigida para desalinhamento e tipo
        y_test_df = pd.read_csv(FILE_PATHS['y_test'], header=None)
        y_oot_df = pd.read_csv(FILE_PATHS['y_oot'], header=None)
        
        # 1. iloc[1:, 0] Remove a primeira linha (cabeçalho implícito)
        # 2. .astype(int) Garante que o target seja int.
        # 3. .reset_index(drop=True) Zera o índice para alinhamento.
        y_test = y_test_df.iloc[1:, 0].rename(TARGET_COL).astype(int).reset_index(drop=True)
        y_oot = y_oot_df.iloc[1:, 0].rename(TARGET_COL).astype(int).reset_index(drop=True)
        
        # Garante que X também tenha o índice limpo para o alinhamento
        X_test = X_test.reset_index(drop=True)
        X_oot = X_oot.reset_index(drop=True)
        
        # --- CORREÇÃO CRÍTICA DO 'ValueError: The feature names should match' ---
        
        # Definir grupos de colunas com base no seu schema
        LAGGED_COLS = [col for col in X_test.columns if 'Lag' in col]
        PRESET_COLS = [col for col in X_test.columns if 'Preset' in col]
        BASE_COLS = [col for col in X_test.columns if col not in LAGGED_COLS and col not in PRESET_COLS]
        
        # 1. Forçar FLOAT nas colunas numéricas base e lags
        cols_to_float = BASE_COLS + LAGGED_COLS
        try:
            X_test[cols_to_float] = X_test[cols_to_float].astype(float)
            X_oot[cols_to_float] = X_oot[cols_to_float].astype(float)
            print("  -> Colunas numéricas forçadas para float.")
        except Exception as e:
            print(f"ERRO: Falha ao converter colunas para float. {e}")
            return None, None, None, None, None

        # 2. Forçar BOOL nas colunas One-Hot Encoded (Presets)
        try:
            X_test[PRESET_COLS] = X_test[PRESET_COLS].astype(bool)
            X_oot[PRESET_COLS] = X_oot[PRESET_COLS].astype(bool)
            print("  -> Colunas Preset forçadas para bool.")
        except Exception as e:
            # Alternativa: se bool falhar, tentar int (que o XGBoost aceita como binário)
            try:
                X_test[PRESET_COLS] = X_test[PRESET_COLS].astype(int)
                X_oot[PRESET_COLS] = X_oot[PRESET_COLS].astype(int)
                print("  -> Colunas Preset forçadas para int (alternativa).")
            except Exception as e_int:
                print(f"ERRO: Falha ao converter Presets para bool/int. {e_int}")
                return None, None, None, None, None
                
        # 3. Checagem final de desalinhamento de tamanho
        if len(X_test) != len(y_test):
            print(f"\nERRO CRÍTICO DE TAMANHO: X_test ({len(X_test)}) e y_test ({len(y_test)}) desalinhados.")
            return None, None, None, None, None
        
        # 4. Carrega o scaler e retorna
        scaler = evaluation_tools.load_model(SCALER_PATH)
        if scaler is None:
            raise FileNotFoundError("Scaler não encontrado.")

        print(f"  -> Verificação de tamanhos OK. Scaler carregado.")
        return X_test, y_test, X_oot, y_oot, scaler
        
    except Exception as e:
        print(f"\nERRO FATAL no carregamento/pré-processamento: {e}")
        return None, None, None, None, None


def preprocess_data(X_data: pd.DataFrame, scaler: any) -> pd.DataFrame:
    """Aplica o scaler (treinado) nos dados de inferência."""
    

    numeric_cols = X_data.select_dtypes(include=np.number).columns.tolist()
    categ_cols = X_data.select_dtypes(include=object).columns.tolist()
    
    # Fit only on training data
    X_data_scaled = X_data.copy()
    X_data_scaled[numeric_cols] = scaler.transform(X_data[numeric_cols])
    
    print("  -> Features standardized using StandardScaler (fitted on X_data).")

    X_data_final = pd.concat([X_data_scaled[numeric_cols], X_data[categ_cols]], axis=1)
    
    return X_data_final



def run_evaluation_pipeline():
    """Executa o pipeline completo de avaliação."""
    
    # 1. Carregamento e Pré-processamento
    X_test_raw, y_test, X_oot_raw, y_oot, scaler = load_data_and_scaler()
    if X_test_raw is None:
        return

    # 2. Aplica o scaler nos dados de teste e OOT
    X_oot_scaled = preprocess_data(X_oot_raw, scaler)
    
    if len(X_test_raw)>0:
        X_test_scaled = preprocess_data(X_test_raw, scaler)
    else:
        print('No test data avaiable - all out of time')
        X_test_scaled = X_oot_scaled
        y_test = y_oot
    
    
    # Tabela para armazenar todos os resultados
    results_list = []
    
    print("\n" + "="*50)
    print("--- 2. INFERÊNCIA E AVALIAÇÃO DE MODELOS ---")
    print("="*50)

    # 3. Itera sobre os modelos
    for model_name, model_path in MODEL_PATHS.items():
        model = evaluation_tools.load_model(model_path)
        if model is None:
            continue

        
            
        # Avaliação no Conjunto de TESTE
        y_proba_test, y_pred_test = evaluation_tools.make_predictions(
            model, X_test_scaled, threshold=0.5 #Generic threshold, will be adjusted later
        )

        CLASSIFICATION_THRESHOLD = evaluation_tools.save_best_treshold(y_test, y_proba_test)
        
        #check min/max probabilities
        print(f'\n--- Modelo: {model_name} ---')
        print(f'Número de amostras Teste: {len(y_test)}')
        print(f'Número de amostras OOT: {len(y_oot)}')
        print(f'Número de positivos Teste: {y_test.sum()}')
        print(f'Número de positivos OOT: {y_oot.sum()}')
        print(f'Número de negativos Teste: {len(y_test) - y_test.sum()}')
        print(f'Número de negativos OOT: {len(y_oot) - y_oot.sum()}')
        print(f'Probabilidades previstas (Teste) - min: {y_proba_test.min():.4f}, max: {y_proba_test.max():.4f}')
        print(f'Predições binárias (Teste) - positivos: {y_pred_test.sum()}, negativos: {len(y_pred_test) - y_pred_test.sum()}')
        

        METRICS_FILE=os.path.join(REPORTS_DIR, 'metrics_summary_test.txt')
        test_metrics = evaluation_tools.calculate_metrics(
            y_test, y_proba_test, y_pred_test, model_name, "Teste", save_path=METRICS_FILE
        )
        results_list.append(test_metrics)
        
        evaluation_tools.plot_confusion_matrix(
            y_test, y_pred_test, model_name, "Teste", save_path=PLOTS_DIR
        )

        # Avaliação no Conjunto OUT-OF-TIME (OOT)
        y_proba_oot, y_pred_oot = evaluation_tools.make_predictions(
            model, X_oot_scaled, threshold=CLASSIFICATION_THRESHOLD
        )
        
        print(f'Probabilidades previstas (OOT) - min: {y_proba_oot.min():.4f}, max: {y_proba_oot.max():.4f}')
        print(f'Predições binárias (OOT) - positivos: {y_pred_oot.sum()}, negativos: {len(y_pred_oot) - y_pred_oot.sum()}')
        print(f'Usando limiar de classificação: {CLASSIFICATION_THRESHOLD}')

        
        METRICS_FILE=os.path.join(REPORTS_DIR, 'metrics_summary_oot.txt')
        oot_metrics = evaluation_tools.calculate_metrics(
            y_oot, y_proba_oot, y_pred_oot, model_name, "OOT", save_path=METRICS_FILE
        )
        results_list.append(oot_metrics)
        
        evaluation_tools.plot_confusion_matrix(
            y_oot, y_pred_oot, model_name, "OOT", save_path=PLOTS_DIR
        )

        #Variable importance plots
        evaluation_tools.plot_feature_importance(
            model, X_test_scaled, model_name, top_n=10, save_path=PLOTS_DIR
        )

    
    # 4. Sumário dos Resultados
    results_df = pd.DataFrame(results_list)
    
    print("\n" + "="*50)
    print("TABELA SUMÁRIO DE MÉTRICAS")
    print("="*50)
    print(results_df[['Model', 'Dataset', 'AUROC', 'Precision', 'Recall', 'F1-Score']])
    
    # 5. Plot Comparativo (OOT é o mais importante)
    evaluation_tools.plot_roc_curve_comparison(
    results_df, MODEL_PATHS, X_oot_scaled, y_oot, save_path=PLOTS_DIR
    )

    
    
    print("\n--- AVALIAÇÃO CONCLUÍDA. Verifique o diretório 'plots/' ---")


if __name__ == '__main__':
    run_evaluation_pipeline()