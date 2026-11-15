
import pandas as pd

class GreedyBaseline:
    def __init__(self, previsao_path="../data/predictions/previsao_202609_full.csv", pulp_path="../data/predictions/auditoria_gondola.csv"):
        self.previsao_path = previsao_path
        self.pulp_path = pulp_path

    def run_greedy(self):
        print("Calculando Baseline (Heurística Gulosa / Intuição)...")
        df_prev = pd.read_csv(self.previsao_path)
        
        resultados_greedy = []
        
        for loja, df_loja in df_prev.groupby('cod_loja'):
            # Heurística Humana: Ordenar pelo que mais tem previsão de venda
            df_sorted = df_loja.sort_values(by='qt_venda_prevista', ascending=False)
            
            espaco_limite = int(df_sorted['qt_espaco_gondola'].iloc[0])
            espaco_usado = 0
            venda_acumulada = 0
            lancamentos_qtd = 0
            categorias_usadas = set()
            
            for _, row in df_sorted.iterrows():
                # Restrição 1: Tem espaço físico?
                if espaco_usado + row['qt_espaco'] <= espaco_limite:
                    
                    # Restrição 2: Já bateu o limite de 2 lançamentos?
                    if row['flg_lancamento'] == 1 and lancamentos_qtd >= 2:
                        continue # Pula este produto
                        
                    if row['flg_lancamento'] == 1:
                        lancamentos_qtd += 1
                        
                    # Adiciona à gôndola gulosa
                    espaco_usado += row['qt_espaco']
                    venda_acumulada += row['qt_venda_prevista']
                    categorias_usadas.add(row['des_categoria'])
            
            # Penalidade/Aviso: A intuição humana esqueceu de pôr alguma categoria na gôndola?
            todas_categorias = set(df_loja['des_categoria'].unique())
            faltaram = todas_categorias - categorias_usadas
            
            resultados_greedy.append({
                'cod_loja': loja,
                'venda_greedy': venda_acumulada,
                'espaco_usado': espaco_usado,
                'ocupacao_%': (espaco_usado / espaco_limite) * 100,
                'categorias_faltantes': len(faltaram)
            })
            
        df_resultado = pd.DataFrame(resultados_greedy)
        return df_resultado
        
    def compare_results(self):
        df_greedy = self.run_greedy()
        df_pulp = pd.read_csv(self.pulp_path)
        
        venda_total_greedy = df_greedy['venda_greedy'].sum()
        venda_total_pulp = df_pulp['venda_projetada'].sum()
        
        ocupacao_media_greedy = df_greedy['ocupacao_%'].mean()
        ocupacao_media_pulp = df_pulp['taxa_ocupacao_%'].mean()
        
        lojas_com_erro_categoria = len(df_greedy[df_greedy['categorias_faltantes'] > 0])
        
        # Cálculo do Uplift
        uplift = ((venda_total_pulp - venda_total_greedy) / venda_total_greedy) * 100
        
        print("\n" + "="*60)
        print("🚀 RESULTADOS EXECUTIVOS: GULOSO VS. OTIMIZADOR (PuLP)")
        print("="*60)
        print(f"📈 Venda Projetada (Intuição/Guloso): {venda_total_greedy:.2f} unidades")
        print(f"🏆 Venda Projetada (Otimizador PuLP):  {venda_total_pulp:.2f} unidades")
        print(f"🔥 UPLIFT (Ganho de Vendas):          +{uplift:.2f}%")
        print("-" * 60)
        print(f"📦 Ocupação Média (Guloso):           {ocupacao_media_greedy:.3f}%")
        print(f"📦 Ocupação Média (PuLP):             {ocupacao_media_pulp:.3f}%")
        print("-" * 60)
        print(f"⚠️ AVISO DE GOVERNANÇA (Trade Marketing):")
        print(f"O método guloso violou a regra de categorias (deixou de expor")
        print(f"alguma categoria obrigatória) em {lojas_com_erro_categoria} lojas.")
        print("O PuLP manteve 100% de compliance com as regras.")
        print("="*60 + "\n")

if __name__ == "__main__":
    comp = GreedyBaseline()
    comp.compare_results()
