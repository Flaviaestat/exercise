import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Tuple, Any, Optional

# --- Importações de SKLEARN e Boosting (Conforme requirements.txt) ---
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline

import xgboost as xgb

from imblearn.over_sampling import SMOTE


# ==============================================================================
# 1. STANDARDIZATION (PADRONIZAÇÃO)
# ==============================================================================

def standardize_features(X_train: pd.DataFrame, X_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardizes numerical features (mean=0, std=1) using a StandardScaler 
    fitted ONLY on the training data.

    Args:
        X_train: Training features DataFrame.
        X_data: Features DataFrame to be transformed (e.g., X_test, X_oot).

    Returns:
        Tuple of (X_train_scaled, X_data_scaled, scaler_object).
    """
    scaler = StandardScaler()
    
    # Identify numerical columns (excluding dummy variables created by OneHotEncoder)
    # Assumes OneHotEncoder was already applied and numerical features are floating point.
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categ_cols = X_train.select_dtypes(include=object).columns.tolist()
    
    # Fit only on training data
    X_train_scaled = X_train.copy()
    X_data_scaled = X_data.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_data_scaled[numeric_cols] = scaler.transform(X_data[numeric_cols])
    
    print("  -> Features standardized using StandardScaler (fitted on X_train).")

    X_train_final = pd.concat([X_train_scaled[numeric_cols], X_train[categ_cols]], axis=1)
    X_data_final = pd.concat([X_data_scaled[numeric_cols], X_data[categ_cols]], axis=1)
    
    return X_train_final, X_data_final


def fit_and_save_scaler(X_train: pd.DataFrame, scaler_path: str) -> StandardScaler:
    """
    Fits o StandardScaler nas colunas numéricas do X_train e salva o objeto scaler.

    Args:
        X_train: Features de treinamento.
        scaler_path: Caminho para salvar o objeto scaler (e.g., 'models/feature_scaler.joblib').

    Returns:
        O objeto StandardScaler fitado.
    """
    scaler = StandardScaler()

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    scaler.fit(X_train[numeric_cols])

    # 2. Salva o scaler fitado
    try:
        joblib.dump(scaler, scaler_path)
        print(f"  -> StandardScaler fitado e salvo em: {scaler_path}")
    except Exception as e:
        print(f"ERRO ao salvar o scaler: {e}")

    return scaler



# ==============================================================================
# 2. BALANCING (BALANCEAMENTO)
# ==============================================================================

def balance_data_smote(X_train: pd.DataFrame, y_train: pd.Series, sampling_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balances the training data using SMOTE (Synthetic Minority Over-sampling Technique).
    
    WARNING: MUST be applied ONLY on training data to prevent data leakage.
    
    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.
        sampling_strategy: Ratio of minority class to majority class after resampling.
                           (e.g., 0.5 means 50% minority instances relative to majority).

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    print(f"\nBalancing data using SMOTE (Target ratio: {sampling_strategy:.0%})...")
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    original_count = y_train.value_counts()
    resampled_count = y_resampled.value_counts()
    
    print(f"  -> Original Target counts:\n{original_count}")
    print(f"  -> Resampled Target counts:\n{resampled_count}")
    
    return X_resampled, y_resampled

# ==============================================================================
# 3. MODEL FITTING WITH HYPERPARAMETER SEARCH (GRID SEARCH)
# ==============================================================================

def train_model_with_gridsearch(
    model_name: str, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    param_grid: Dict[str, List[Any]],
    target_col: str,
    cv_folds: int = 5
) -> Any:
    """
    Trains a classification model using GridSearchCV with TimeSeriesSplit.
    
    Applies class weights for initial imbalance handling.

    Args:
        model_name: Name of the model ('LogisticRegression' or 'XGBoost').
        X_train: Training features DataFrame.
        y_train: Training target Series.
        param_grid: Dictionary of hyperparameters to search.
        target_col: The name of the target column (e.g., 'Target').
        cv_folds: Number of splits for TimeSeriesSplit.

    Returns:
        The best fitted model (classifier object).
    """
    print(f"\n--- FITTING MODEL: {model_name} ---")

    # 1. Define Model and Class Weights
    if model_name == 'LogisticRegression':
        classifier = LogisticRegression(solver='liblinear', random_state=42, 
                                        class_weight='balanced', max_iter=500)
    elif model_name == 'MLPClassifier':
        classifier = MLPClassifier(random_state=42, 
                                   max_iter=500)
    elif model_name == 'SVM':
        classifier = SVC(random_state=42, class_weight='balanced', probability=True)
    elif model_name == 'XGBoost':
        # scale_pos_weight is XGBoost's equivalent of class_weight
        # Calculated as: (Total Negative Samples) / (Total Positive Samples)
        neg_count = y_train.value_counts().get(0, 1)
        pos_count = y_train.value_counts().get(1, 1)
        scale_pos_weight_val = neg_count / pos_count
        
        classifier = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss', 
            use_label_encoder=False, 
            random_state=42,
            scale_pos_weight=scale_pos_weight_val # Handling imbalance
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported here.")

    # 2. Define Cross-Validation Strategy (Time Series Split)
    # TimeSeriesSplit ensures that the model is always validated on future data.
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    # 3. Setup GridSearch
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring='f1', # Use F1-Score due to class imbalance
        cv=tscv,
        verbose=2,
        n_jobs=-1  # Use all available cores
    )

    # 4. Fit the model
    grid_search.fit(X_train, y_train)

    print(f"\n  -> Best F1-Score: {grid_search.best_score_:.4f}")
    print(f"  -> Best Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_


# ==============================================================================
# 4. SAVE MODEL
# ==============================================================================

def save_model(model: Any, filename: str) -> None:
    """
    Saves the trained model object to a file using joblib.

    Args:
        model: The fitted model object.
        filename: Path and name of the file (e.g., 'models/best_xgb.joblib').
    """
    try:
        joblib.dump(model, filename)
        print(f"\nModel successfully saved to: {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")
