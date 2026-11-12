
import pandas as pd
import pulp
import os

class GondolaOptimization:
    def __init__(self, input_path="../data/predictions/previsao_202609_full.csv", output_path="../data/predictions/gondola_202609.csv"):
        self.input_path = input_path
        self.output_path = output_path
        
    def load_data(self):
        print(f"Carregando previsões para otimização... (PuLP versão {pulp.__version__})")
        self.df = pd.read_csv(self.input_path)
        
    def optimize_store(self, df_loja):
        # 1. Definir o problema de Maximização
        prob = pulp.LpProblem("Otimizacao_Gondola", pulp.LpMaximize)
        skus = df_loja['cod_sku'].tolist()
        
        # 2. Variáveis de decisão (API compatível com PuLP 4.x e 3.x)
        pulp_v4 = int(pulp.__version__.split('.')[0]) >= 4
        if pulp_v4:
            x = prob.add_variable_dicts("SKU", skus, cat='Binary')
        else:
            x = pulp.LpVariable.dicts("SKU", skus, cat='Binary')
        
        # 3. Função Objetivo: Maximizar a soma das vendas previstas
        # Convertendo para float nativo para evitar conflitos de numpy types
        vendas = {k: float(v) for k, v in zip(skus, df_loja['qt_venda_prevista'])}
        prob += pulp.lpSum([vendas[i] * x[i] for i in skus]), "Max_Vendas"
        
        # 4. Restrições
        espacos = {k: float(v) for k, v in zip(skus, df_loja['qt_espaco'])}
        
        # CORREÇÃO AQUI: Converter numpy.int64 para int nativo do Python
        espaco_gondola = int(df_loja['qt_espaco_gondola'].iloc[0])
        
        prob += pulp.lpSum([espacos[i] * x[i] for i in skus]) <= espaco_gondola, "Restricao_Espaco"
        
        lancamentos = {k: int(v) for k, v in zip(skus, df_loja['flg_lancamento'])}
        prob += pulp.lpSum([lancamentos[i] * x[i] for i in skus]) <= 2, "Restricao_Lancamentos"
        
        categorias = df_loja['des_categoria'].unique()
        skus_por_categoria = {cat: df_loja[df_loja['des_categoria'] == cat]['cod_sku'].tolist() for cat in categorias}
        for cat in categorias:
            prob += pulp.lpSum([x[i] for i in skus_por_categoria[cat]]) >= 1, f"Restricao_Categoria_{cat}"
            
        # 5. Resolver o problema silenciosamente
        try:
            if pulp_v4:
                prob.solve(pulp.COIN_CMD(msg=False))
            else:
                prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception as e:
            prob.solve()
        
        # 6. Coletar os SKUs selecionados
        skus_selecionados = [i for i in skus if x[i].varValue == 1.0]
        
        return skus_selecionados

    def run_optimization(self):
        print("\n=== Iniciando Otimização de Gôndola (Problema da Mochila) ===")
        resultados = []
        lojas = self.df['cod_loja'].unique()
        
        for i, loja in enumerate(lojas):
            if i % 10 == 0:
                print(f"Otimizando loja {i+1}/{len(lojas)}...")
                
            df_loja = self.df[self.df['cod_loja'] == loja]
            skus_selecionados = self.optimize_store(df_loja)
            
            for sku in skus_selecionados:
                resultados.append({
                    'cod_loja': loja,
                    'cod_sku': sku
                })
                
        df_gondola = pd.DataFrame(resultados)
        
        # Validação cruzada rápida
        df_verificacao = df_gondola.merge(self.df[['cod_loja', 'cod_sku', 'qt_espaco', 'qt_espaco_gondola']], on=['cod_loja', 'cod_sku'], how='left')
        espaco_utilizado = df_verificacao.groupby('cod_loja')['qt_espaco'].sum()
        limites = df_verificacao.groupby('cod_loja')['qt_espaco_gondola'].first()
        
        violacoes = espaco_utilizado > limites
        if violacoes.any():
            print(f"\nAVISO: {violacoes.sum()} lojas violaram o limite de espaço!")
        else:
            print("\nSucesso! Todas as lojas respeitaram o limite de espaço e restrições.")
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df_gondola.to_csv(self.output_path, index=False)
        print(f"Arquivo da gôndola (Opcional 1) salvo em: {self.output_path}")
        
        return df_gondola

if __name__ == "__main__":
    optimizer = GondolaOptimization()
    optimizer.load_data()
    df_gondola = optimizer.run_optimization()
