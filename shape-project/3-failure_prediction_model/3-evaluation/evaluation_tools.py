import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import io

from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve
)

# ==============================================================================
# 1. LOAD MODEL
# ==============================================================================

def load_model(filename: str) -> Any:
    """
    Carrega um objeto (modelo ou scaler) do disco usando joblib.

    Args:
        filename: Caminho e nome do arquivo (e.g., 'models/logistic_best.joblib').

    Returns:
        O objeto carregado (modelo treinado ou scaler).
    """
    try:
        model = joblib.load(filename)
        print(f"  -> Objeto carregado com sucesso de: {filename}")
        return model
    except Exception as e:
        print(f"ERRO ao carregar o arquivo {filename}: {e}")
        return None

# ==============================================================================
# 2. INFERENCE (INFERÊNCIA)
# ==============================================================================

def make_predictions(model: Any, X_data: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faz previsões de probabilidade e classe para um dado modelo.

    Args:
        model: O modelo treinado (LogisticRegression ou XGBoost).
        X_data: Features de teste ou OOT.
        threshold: Limiar de corte para classificar (prob >= threshold -> 1).

    Returns:
        Tuple de (probabilidades, previsões binárias).
    """
    if model is None:
        raise ValueError("Modelo não pode ser None.")
        
    # Previsões de probabilidade para a classe positiva (1)
    y_proba = model.predict_proba(X_data)[:, 1]
    
    # Previsões binárias usando o limiar
    y_pred = (y_proba >= threshold).astype(int)
    
    return y_proba, y_pred


# ==============================================================================
# 3. METRICS CALCULATION (CÁLCULO DE MÉTRICAS)
# ==============================================================================

def save_best_treshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """
    Calcula, reporta e Opcionalmente salva métricas de desempenho chave para a classificação,
    incluindo a busca pelo threshold ideal que maximiza o F1-Score.

    Args:
        y_true: Valores verdadeiros (target).
        y_proba: Probabilidades preditas (para AUC e busca de threshold).
    Returns:
        Dicionário com as métricas calculadas.
    """
    
    # ----------------------------------------------------
    # 1. ENCONTRAR O THRESHOLD ÓTIMO (Maximiza F1-Score)
    # ----------------------------------------------------
    
    # Gerar a curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return thresholds[np.argmax(tpr - fpr)]




def calculate_metrics(y_true: pd.Series, y_proba: np.ndarray, y_pred: np.ndarray, 
                      model_name: str, dataset_name: str, save_path: Optional[str] = None) -> Dict[str, float]:
    """
    Calcula, reporta e Opcionalmente salva métricas de desempenho chave para a classificação,
    incluindo a busca pelo threshold ideal que maximiza o F1-Score.

    Args:
        y_true: Valores verdadeiros (target).
        y_proba: Probabilidades preditas (para AUC e busca de threshold).
        y_pred: Classes binárias preditas (usadas para o threshold inicial de 0.5).
        model_name: Nome do modelo.
        dataset_name: Nome do conjunto (Teste, OOT).
        save_path: Caminho para salvar o arquivo de texto.

    Returns:
        Dicionário com as métricas calculadas.
    """
    
    # ----------------------------------------------------
    # 1. ENCONTRAR O THRESHOLD ÓTIMO (Maximiza F1-Score)
    # ----------------------------------------------------
    
    # Gerar a curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Calcular F1-Score para todos os thresholds
    f1_scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    
    # Encontrar o threshold que maximiza o F1-Score
    best_f1_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_index]
    
    # Recalcular as previsões e métricas usando o threshold ótimo
    y_pred_best = (y_proba >= best_threshold).astype(int)
    
    # ----------------------------------------------------
    # 2. CÁLCULO DAS MÉTRICAS (Usando o threshold OTIMIZADO)
    # ----------------------------------------------------
    
    metrics = {
        'Model': model_name,
        'Dataset': dataset_name,
        'Threshold_Default_Used': 0.5, # Armazena o threshold padrão usado no fit original
        'AUROC': roc_auc_score(y_true, y_proba),
        'Best_Threshold': best_threshold, # O novo threshold
        'Precision': precision_score(y_true, y_pred_best, zero_division=0),
        'Recall': recall_score(y_true, y_pred_best, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred_best, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred_best)
    }
    
    # ----------------------------------------------------
    # 3. FORMATAR SAÍDA PARA CONSOLE E ARQUIVO
    # ----------------------------------------------------
    
    output = io.StringIO()
    output.write(f"\n--- Métricas Otimizadas: {model_name} em {dataset_name} ---\n")
    output.write(f"AUROC (Capac. Discriminativa): {metrics['AUROC']:.4f}\n")
    output.write(f"BEST THRESHOLD (Max F1): {metrics['Best_Threshold']:.4f}\n")
    output.write(f"--------------------------------------------------\n")
    output.write(f"Recall (Capac. p/ Capturar Falha): {metrics['Recall']:.4f} (com novo threshold)\n")
    output.write(f"Precision (Confiabilidade): {metrics['Precision']:.4f} (com novo threshold)\n")
    output.write(f"F1-Score (Otimizado): {metrics['F1-Score']:.4f}\n")
    output.write("-" * 50 + "\n")
    
    report_text = output.getvalue()
    
    # 4. IMPRIMIR E SALVAR
    print(report_text)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'a') as f:
            f.write(report_text)
        print(f"  -> Métricas salvas em: {save_path}")

    # Retorna as métricas OTIMIZADAS
    return metrics



# ==============================================================================
# 4. PLOTTING (GRÁFICOS)
# ==============================================================================

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_name: str, dataset_name: str, save_path: str = 'plots/') -> None:
    """
    Gera e salva a matriz de confusão.

    Args:
        y_true: Valores verdadeiros (target).
        y_pred: Classes binárias preditas.
        model_name: Nome do modelo.
        dataset_name: Nome do conjunto (Teste, OOT).
        save_path: Diretório para salvar o plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Falha (0)', 'Falha (1)'], 
                yticklabels=['Não Falha (0)', 'Falha (1)'])
    
    plt.title(f'Matriz de Confusão: {model_name} em {dataset_name}')
    plt.ylabel('Valores Reais')
    plt.xlabel('Previsões')
    
    # Salvar o arquivo
    filename = f'{model_name}_{dataset_name}_CM.png'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
    print(f"  -> Matriz de Confusão salva em: {os.path.join(save_path, filename)}")

def plot_roc_curve_comparison(results_df: pd.DataFrame, model_paths: Dict[str, str], X_data: pd.DataFrame, y_data: pd.Series, save_path: str = 'plots/'):
    """
    Compara curvas ROC para diferentes modelos.
    """
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.50)')

    for model_name, path in model_paths.items():
        try:
            model = load_model(path)
            y_proba, _ = make_predictions(model, X_data, threshold=0.5)
            fpr, tpr, _ = roc_curve(y_data, y_proba)
            auc_score = roc_auc_score(y_data, y_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
        except Exception as e:
            print(f"Aviso: Não foi possível plotar curva ROC para {model_name}: {e}")

    plt.xlabel('False Positive Rate (Taxa de Falsos Positivos)')
    plt.ylabel('True Positive Rate (Taxa de Verdadeiros Positivos / Recall)')
    plt.title(f'Curva ROC Comparativa no Conjunto OOT')
    plt.legend(loc="lower right")
    
    # Salvar o arquivo
    filename = 'ROC_Comparison_OOT.png'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
    print(f"  -> Curva ROC Comparativa salva em: {os.path.join(save_path, filename)}")



def plot_feature_importance(model: Any, X_data: pd.DataFrame, model_name: str, 
                            top_n: int = 15, save_path: str = 'plots/') -> None:
    """
    Calcula, plota e salva a importância das features para LogisticRegression ou XGBoost.

    Args:
        model: O modelo treinado.
        X_data: DataFrame de features (usado para obter os nomes das colunas).
        model_name: Nome do modelo ('LogisticRegression' ou 'XGBoost').
        top_n: Número de features mais importantes a serem exibidas.
        save_path: Diretório para salvar o plot.
    """
    
    feature_names = X_data.columns.tolist()
    
    # 1. Obter a importância baseada no modelo
    if model_name == 'LogisticRegression':
        # Para Logística: Usamos o valor ABSOLUTO dos coeficientes.
        # Isso só é válido porque padronizamos as features.
        importance = np.abs(model.coef_[0])
        y_label = 'Importância (Coeficiente Absoluto)'
    
    elif model_name == 'XGBoost':
        # Para XGBoost: Usamos o feature_importances_ (que é o 'Gain' por padrão no sklearn API)
        importance = model.feature_importances_
        y_label = 'Importância (Gain)'
        
    else:
        print(f"Aviso: Plot de importância não suportado para o modelo {model_name}.")
        return

    # 2. Criar a Série Pandas para organização e ranking
    importance_series = pd.Series(importance, index=feature_names)
    top_importance = importance_series.nlargest(top_n)

    # 3. Plotagem
    plt.figure(figsize=(10, top_n * 0.4))
    
    # Gráfico de barras horizontal (melhor para muitos features)
    top_importance.sort_values().plot(kind='barh', color='skyblue')
    
    plt.title(f'Top {top_n} Importância de Features: {model_name}')
    plt.xlabel(y_label)
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Salvar o arquivo
    filename = f'{model_name}_Feature_Importance.png'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
    plt.close()
    print(f"  -> Gráfico de Importância salvo em: {os.path.join(save_path, filename)}")
