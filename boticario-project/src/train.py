import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os
import glob

class ModelTraining:
    def __init__(self, train_path="../data/processed_data/df_train.csv", test_path_pattern="../data/processed_data/df_test_*.csv"):
        self.train_path = train_path
        
        test_files = glob.glob(test_path_pattern)
        if not test_files:
            raise FileNotFoundError("Nenhum arquivo de teste encontrado. Rode o features.py primeiro.")
        self.test_path = max(test_files, key=os.path.getctime)
        
    def load_datasets(self):
        print("Carregando datasets para modelagem...")
        self.df_train = pd.read_csv(self.train_path)
        self.df_test = pd.read_csv(self.test_path)
        
        self.max_nr_ordem = self.df_train["nr_ordem"].max()
        self.target_ciclo = self.df_test["cod_ciclo"].iloc[0]
        
        # Removidos cod_loja e cod_sku do casting (não serão features)
        cat_cols = ["tp_perfil", "des_regiao", "des_categoria"]
        for col in cat_cols:
            if col in self.df_train.columns:
                self.df_train[col] = self.df_train[col].astype("category")
            if col in self.df_test.columns:
                self.df_test[col] = self.df_test[col].astype("category")
            
    def train_validate(self):
        print(f"\n=== Validacao Temporal (Treino: <= {self.max_nr_ordem - 1}, Teste: {self.max_nr_ordem}) ===")
        
        # Removidos cod_loja e cod_sku das features preditivas
        features = [
            "flg_campanha", "tp_perfil", "des_regiao", 
            "qt_espaco_gondola", "des_categoria", "qt_espaco", "vlr_preco_tabela", 
            "flg_lancamento", "lag_1", "lag_2", "lag_3", "rolling_mean_3", "diff_1", "rolling_std_3"
        ]
        target = "qt_venda"
        
        train = self.df_train[self.df_train["nr_ordem"] < self.max_nr_ordem]
        val = self.df_train[self.df_train["nr_ordem"] == self.max_nr_ordem]
        
        X_train, y_train = train[features], train[target]
        X_val, y_val = val[features], val[target]
        
        # Treinando LightGBM
        print("Treinando LightGBM...")
        model_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds_lgb = model_lgb.predict(X_val)
        mae_lgb = mean_absolute_error(y_val, preds_lgb)

        # Treinando XGBoost
        print("Treinando XGBoost...")
        model_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, enable_categorical=True)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds_xgb = model_xgb.predict(X_val)
        mae_xgb = mean_absolute_error(y_val, preds_xgb)
        
        # ==================== Exportando Resultados e Feature Importance ====================
        df_val_results = val[["cod_loja", "cod_sku", "qt_venda"]].copy()
        df_val_results["pred_lgbm"] = preds_lgb
        df_val_results["pred_xgb"] = preds_xgb
        
        val_path = f"../data/validation/val_ciclo_{self.max_nr_ordem}_resultados.csv"
        os.makedirs(os.path.dirname(val_path), exist_ok=True)
        df_val_results.to_csv(val_path, index=False)
        
        # Exportando Importancia das Variaveis para o Evaluation plotar

        df_imp = pd.DataFrame({
            'Feature': features,
            # Forçamos o LightGBM a usar 'gain' em vez do padrão 'split'
            'LGBM_Importance': model_lgb.booster_.feature_importance(importance_type='gain'),
            'XGB_Importance': model_xgb.feature_importances_
        })
        imp_path = f"../data/validation/feature_importance_{self.max_nr_ordem}.csv"
        df_imp.to_csv(imp_path, index=False)
        print(f"\nResultados e Feature Importance salvos em: ../data/validation/")

        # Define Vencedor
        if mae_lgb <= mae_xgb:
            print(f"\n>>> Vencedor: LightGBM (MAE: {mae_lgb:.2f})")
            best_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
            model_name = "LightGBM"
        else:
            print(f"\n>>> Vencedor: XGBoost (MAE: {mae_xgb:.2f})")
            best_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, enable_categorical=True)
            model_name = "XGBoost"

        return best_model, features, model_name
        
    def final_forecast(self, model, features, model_name):
        print(f"\n=== Treinando Modelo Final ({model_name}) e Gerando Previsao para {self.target_ciclo} ===")
        
        X_train = self.df_train[features]
        y_train = self.df_train["qt_venda"]
        X_test = self.df_test[features]
        
        model.fit(X_train, y_train)
        
        self.df_test["qt_venda_prevista"] = model.predict(X_test)
        self.df_test["qt_venda_prevista"] = self.df_test["qt_venda_prevista"].clip(lower=0)
        
        output_cols = ["cod_loja", "cod_sku", "cod_ciclo", "qt_venda_prevista"]
        df_submission = self.df_test[output_cols].copy()
        
        pred_path = f"../data/predictions/previsao_{self.target_ciclo}.csv"
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        df_submission.to_csv(pred_path, index=False)
        
        output_full = f"../data/predictions/previsao_{self.target_ciclo}_full.csv"
        self.df_test.to_csv(output_full, index=False)
        
        return df_submission

if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.load_datasets()
    best_model, features, model_name = trainer.train_validate()
    final_preds = trainer.final_forecast(best_model, features, model_name)
