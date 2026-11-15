import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    import optuna
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import PredefinedSplit, validation_curve
    from sklearn.metrics import mean_squared_error
except ImportError:
    print("Por favor, instale as dependências: pip install optuna lightgbm scikit-learn matplotlib")

class TreinamentoTurbo:
    def __init__(self, data_path="../data/processed_data/df_train.csv"):
        self.data_path = data_path
        self.ciclo_validacao = 202608 # Último ciclo conhecido para validar
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset carregado com {len(self.df)} linhas.")
        except FileNotFoundError:
            print(f"Erro: Arquivo {self.data_path} não encontrado.")
            return

        # 1. TRATAMENTO DE CATEGÓRICAS PREDITIVAS
        # Removidos cod_loja e cod_sku (IDs não devem ser features para forçar generalização)
        colunas_categoricas = ['tp_perfil', 'des_regiao', 'des_categoria']
        for col in colunas_categoricas:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
                
        # 2. ISOLAR COLUNAS (Adicionando os IDs na lista de exclusão)
        colunas_ignoradas = ['qt_venda', 'cod_ciclo', 'des_ciclo', 'des_sku', 'cod_loja', 'cod_sku']
        self.features = [col for col in self.df.columns if col not in colunas_ignoradas]
        
        # Separação Temporal Estrita (Prevenindo Data Leakage)
        self.treino = self.df[self.df['cod_ciclo'] < self.ciclo_validacao].copy()
        self.validacao = self.df[self.df['cod_ciclo'] == self.ciclo_validacao].copy()
        
        self.X_train = self.treino[self.features]
        self.y_train = self.treino['qt_venda']
        self.X_val = self.validacao[self.features]
        self.y_val = self.validacao['qt_venda']

    def objective(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMRegressor(**param)
        model.fit(self.X_train, self.y_train)
        
        preds = model.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, preds))
        return rmse

    def rodar_optuna(self, n_trials=20):
        print("\n=== Iniciando Busca Bayesiana com Optuna ===")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print("🔥 Melhores Hiperparâmetros encontrados:")
        for key, value in study.best_params.items():
            print(f"   - {key}: {value}")
        print(f"📉 Melhor RMSE na Validação: {study.best_value:.4f}")
        
        self.best_params = study.best_params
        return study

    def gerar_curva_validacao(self):
        print("\n=== Gerando Curva de Validação (Validation Curve) ===")
        indices_split = np.where(self.df['cod_ciclo'] == self.ciclo_validacao, 0, -1)
        ps = PredefinedSplit(indices_split)
        
        X_full = self.df[self.features]
        y_full = self.df['qt_venda']
        
        param_range = np.arange(2, 22, 2)
        base_params = self.best_params.copy() if hasattr(self, 'best_params') else {'random_state': 42, 'verbose': -1}
        if 'max_depth' in base_params:
            del base_params['max_depth']
            
        model = LGBMRegressor(**base_params)
        
        train_scores, test_scores = validation_curve(
            model, X_full, y_full, 
            param_name="max_depth", 
            param_range=param_range,
            cv=ps,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.title("Curva de Validação: Identificando Overfitting (LightGBM)")
        plt.xlabel("Profundidade da Árvore (max_depth)")
        plt.ylabel("Erro (RMSE) - Quanto menor, melhor")
        plt.ylim(0, max(max(train_scores_mean), max(test_scores_mean)) * 1.1)
        
        plt.plot(param_range, train_scores_mean, label="Erro no Treino", color="darkorange", lw=2, marker='o')
        plt.plot(param_range, test_scores_mean, label="Erro na Validação (Ciclo Futuro)", color="navy", lw=2, marker='s')
        plt.axvline(x=self.best_params.get('max_depth', 6), color='red', linestyle='--', label=f"Profundidade Ótima (Optuna)")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        os.makedirs("../validation/validation_curve/", exist_ok=True)
        img_path = "../validation/validation_curve/validation_curve_max_depth.png"
        plt.savefig(img_path, dpi=300)
        print(f"✅ Gráfico salvo com sucesso em: {img_path}")

    def gerar_previsoes_finais(self, inference_path="../data/processed_data/df_test.csv", output_path="../data/predictions/previsao_202609_full.csv"):
        print("\n=== Gerando Previsões Finais com o Melhor Modelo ===")
        X_full = self.df[self.features]
        y_full = self.df['qt_venda']
        
        best_model = LGBMRegressor(**self.best_params)
        best_model.fit(X_full, y_full)
        
        try:
            df_inf = pd.read_csv(inference_path)
        except FileNotFoundError:
            print(f"Erro: {inference_path} não encontrado.")
            return
            
        colunas_categoricas = ['tp_perfil', 'des_regiao', 'des_categoria']
        for col in colunas_categoricas:
            if col in df_inf.columns:
                df_inf[col] = df_inf[col].astype('category')
                
        X_inf = df_inf[self.features]
        
        df_inf['qt_venda_prevista'] = np.ceil(best_model.predict(X_inf)).astype(int)
        df_inf['qt_venda_prevista'] = np.where(df_inf['qt_venda_prevista'] < 0, 0, df_inf['qt_venda_prevista'])
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_inf.to_csv(output_path, index=False)
        print(f"✅ Arquivo de previsões atualizado para o PuLP/FastAPI em: {output_path}")

if __name__ == "__main__":
    turbo = TreinamentoTurbo()
    turbo.load_data()
    if hasattr(turbo, 'df'):
        turbo.rodar_optuna(n_trials=15)
        turbo.gerar_curva_validacao()
        turbo.gerar_previsoes_finais()
