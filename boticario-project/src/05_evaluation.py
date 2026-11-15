
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os
import glob

class ModelEvaluation:
    def __init__(self, val_path_pattern="../data/validation/val_ciclo_*_resultados.csv", imp_path_pattern="../data/validation/feature_importance_*.csv", output_dir="../data/validation/"):
        val_files = glob.glob(val_path_pattern)
        imp_files = glob.glob(imp_path_pattern)
        
        if not val_files or not imp_files:
            raise FileNotFoundError("Arquivos de validacao nao encontrados. Rode o train.py primeiro.")
            
        self.val_path = max(val_files, key=os.path.getctime)
        self.imp_path = max(imp_files, key=os.path.getctime)
        self.output_dir = output_dir
        
    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def weighted_mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        sum_true = np.sum(y_true)
        if sum_true == 0:
            return np.nan
        return (np.sum(np.abs(y_true - y_pred)) / sum_true) * 100
        
    def wrmsse(self, df_val, df_train_full, pred_col, ciclo_val):
        """Calcula o Weighted Root Mean Squared Scaled Error (WRMSSE) inspirado no M5."""
        # 1. Dados de Treino (Exclui a janela de validacao)
        df_train_only = df_train_full[df_train_full["nr_ordem"] < ciclo_val].copy()
        
        # 2. Pesos baseados na receita do ultimo ciclo de treino (ciclo_val - 1)
        df_last_train = df_train_only[df_train_only["nr_ordem"] == ciclo_val - 1].copy()
        df_last_train["receita"] = df_last_train["qt_venda"] * df_last_train["vlr_preco_tabela"]
        total_receita = df_last_train["receita"].sum()
        
        if total_receita > 0:
            df_last_train["weight"] = df_last_train["receita"] / total_receita
        else:
            df_last_train["weight"] = 1.0 / len(df_last_train)
            
        weights = df_last_train.set_index(["cod_loja", "cod_sku"])["weight"]
        
        # 3. Denominador RMSSE (MSE do modelo Naive In-Sample)
        # O naive forecast seria prever o lag_1. O erro in-sample eh (qt_venda - lag_1)^2
        df_train_only["naive_sq_error"] = (df_train_only["qt_venda"] - df_train_only["lag_1"])**2
        denom = df_train_only.groupby(["cod_loja", "cod_sku"])["naive_sq_error"].mean()
        
        # 4. Calculo do RMSSE e aplicacao dos pesos
        df_calc = df_val[["cod_loja", "cod_sku", "qt_venda", pred_col]].copy()
        df_calc = df_calc.join(weights, on=["cod_loja", "cod_sku"], how="left").fillna({"weight": 0})
        df_calc = df_calc.join(denom.rename("denom"), on=["cod_loja", "cod_sku"], how="left")
        
        # Substitui denominador 0 por 1 para evitar divisao por zero (series perfeitamente constantes no treino)
        df_calc["denom"] = df_calc["denom"].replace(0, 1).fillna(1)
        
        df_calc["mse_val"] = (df_calc["qt_venda"] - df_calc[pred_col])**2
        df_calc["rmsse"] = np.sqrt(df_calc["mse_val"] / df_calc["denom"])
        
        wrmsse_value = (df_calc["weight"] * df_calc["rmsse"]).sum()
        return wrmsse_value

    def generate_metrics(self):
        print("\n=== Metricas Globais ===")
        df_val = pd.read_csv(self.val_path)
        
        # Recupera as features de volta para analise segmentada (merge com o df_train original)
        ciclo_val = int(self.val_path.split("_ciclo_")[1].split("_")[0])
        df_train_full = pd.read_csv("../data/processed_data/df_train.csv")
        df_val_features = df_train_full[df_train_full["nr_ordem"] == ciclo_val]
        df_val = df_val.merge(df_val_features[["cod_loja", "cod_sku", "tp_perfil", "des_regiao", "des_categoria", "flg_campanha", "flg_lancamento"]], on=["cod_loja", "cod_sku"], how="left")
        
        y_true = df_val["qt_venda"]
        pred_lgb = df_val["pred_lgbm"]
        pred_xgb = df_val["pred_xgb"]
        
        metrics = []
        for model_name, preds, col_name in zip(["LightGBM", "XGBoost"], [pred_lgb, pred_xgb], ["pred_lgbm", "pred_xgb"]):
            mae = mean_absolute_error(y_true, preds)
            rmse = root_mean_squared_error(y_true, preds)
            mape = self.mean_absolute_percentage_error(y_true, preds)
            wmape = self.weighted_mape(y_true, preds)
            wrmsse_val = self.wrmsse(df_val, df_train_full, col_name, ciclo_val)
            
            metrics.append({
                "Modelo": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "WMAPE (%)": wmape,
                "WRMSSE": wrmsse_val
            })
            
        df_metrics = pd.DataFrame(metrics)
        metrics_path = os.path.join(self.output_dir, "comparacao_metricas.csv")
        df_metrics.to_csv(metrics_path, index=False)
        print(df_metrics.to_string(index=False))
        return df_val

    def generate_segmented_metrics(self, df_val):
        print("\n=== Gerando Metricas Segmentadas (LightGBM) ===")
        segments = ["tp_perfil", "des_regiao", "des_categoria", "flg_campanha", "flg_lancamento"]
        
        with open(os.path.join(self.output_dir, "metricas_segmentadas.txt"), "w") as f:
            for seg in segments:
                print(f"\n--- Erro por {seg} ---")
                f.write(f"\n--- Erro por {seg} ---\n")
                
                res = []
                for name, group in df_val.groupby(seg):
                    mae = mean_absolute_error(group["qt_venda"], group["pred_lgbm"])
                    wmape = self.weighted_mape(group["qt_venda"], group["pred_lgbm"])
                    volume = group["qt_venda"].sum()
                    res.append((name, volume, mae, wmape))
                
                df_seg = pd.DataFrame(res, columns=[seg, "Volume_Real", "MAE", "WMAPE (%)"]).round(2)
                df_seg = df_seg.sort_values("Volume_Real", ascending=False)
                
                print(df_seg.to_string(index=False))
                f.write(df_seg.to_string(index=False) + "\n")
                
        print(f"\nMetricas segmentadas salvas em: metricas_segmentadas.txt")

    def plot_residuals(self, df_val):
        print(f"\nGerando plot de Distribuicao dos Residuos...")
        df_val["residuos_lgbm"] = df_val["pred_lgbm"] - df_val["qt_venda"]
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_val["residuos_lgbm"], bins=50, kde=True, color="purple")
        
        plt.axvline(x=0, color="red", linestyle="--", label="Erro Zero")
        plt.title("Distribuicao dos Residuos (Previsto - Real) - LightGBM")
        plt.xlabel("Residuo (> 0 Superestimou | < 0 Subestimou)")
        plt.ylabel("Frequencia")
        plt.legend()
        
        plot_path = os.path.join(self.output_dir, "plot_residuos.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_feature_importance(self):
        print(f"Gerando plot de Feature Importance...")
        df_imp = pd.read_csv(self.imp_path)
        
        df_imp["LGBM_Norm"] = df_imp["LGBM_Importance"] / df_imp["LGBM_Importance"].sum()
        df_imp["XGB_Norm"] = df_imp["XGB_Importance"] / df_imp["XGB_Importance"].sum()
        
        df_imp["Mean_Imp"] = (df_imp["LGBM_Norm"] + df_imp["XGB_Norm"]) / 2
        df_imp = df_imp.sort_values(by="Mean_Imp", ascending=True)
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y = np.arange(len(df_imp))
        height = 0.35
        
        ax.barh(y - height/2, df_imp["LGBM_Norm"], height, label="LightGBM", color="blue")
        ax.barh(y + height/2, df_imp["XGB_Norm"], height, label="XGBoost", color="orange")
        
        ax.set_yticks(y)
        ax.set_yticklabels(df_imp["Feature"])
        ax.set_xlabel("Importancia Normalizada")
        ax.set_title("Comparacao de Feature Importance (LGBM vs XGBoost)")
        ax.legend()
        
        plt.tight_layout()
        imp_plot_path = os.path.join(self.output_dir, "plot_feature_importance.png")
        plt.savefig(imp_plot_path, dpi=300, bbox_inches="tight")
        print(f"Plots salvos na pasta de validacao.")
        plt.close()

    def plot_actual_vs_predicted(self, df_val, num_lojas=4):
        print(f"\nGerando plots Previsto x Real para {num_lojas} lojas aleatorias...")
        # Seleciona algumas lojas aleatorias que tiveram vendas
        lojas_com_venda = df_val[df_val["qt_venda"] > 0]["cod_loja"].unique()
        lojas_amostra = np.random.choice(lojas_com_venda, min(num_lojas, len(lojas_com_venda)), replace=False)
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, loja in enumerate(lojas_amostra):
            df_loja = df_val[df_val["cod_loja"] == loja].copy()
            # Ordena por qt_venda para o plot ficar mais legivel
            df_loja = df_loja.sort_values(by="qt_venda", ascending=False).reset_index(drop=True)
            # Limita a 20 produtos para nao poluir
            df_loja = df_loja.head(20)
            
            ax = axes[i]
            x = np.arange(len(df_loja))
            width = 0.25
            
            ax.bar(x - width, df_loja["qt_venda"], width, label="Real", color="gray")
            ax.bar(x, df_loja["pred_lgbm"], width, label="LightGBM", color="blue")
            ax.bar(x + width, df_loja["pred_xgb"], width, label="XGBoost", color="orange")
            
            ax.set_title(f"Loja: {loja} (Top 20 Skus)")
            ax.set_xticks(x)
            ax.set_xticklabels(df_loja["cod_sku"], rotation=90, fontsize=8)
            ax.set_ylabel("Quantidade")
            if i == 0:
                ax.legend()
                
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "plot_previsto_x_real.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot salvo em: {plot_path}")
        plt.close()


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    df_val = evaluator.generate_metrics()
    evaluator.generate_segmented_metrics(df_val)
    evaluator.plot_residuals(df_val)
    evaluator.plot_feature_importance()
    evaluator.plot_actual_vs_predicted(df_val, num_lojas=4)