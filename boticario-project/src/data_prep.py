import pandas as pd
import os

class DataPrep:
    def __init__(self, filepath='../data/raw_data/base-sortimento-campanha.xlsx'):
        self.filepath = filepath
        
    def load_data(self):
        """Carrega as 4 abas principais da base de dados."""
        print("Carregando os dados brutos...")
        self.tb_loja = pd.read_excel(self.filepath, sheet_name='tb_loja')
        self.tb_sku = pd.read_excel(self.filepath, sheet_name='tb_sku')
        self.tb_ciclo = pd.read_excel(self.filepath, sheet_name='tb_ciclo')
        self.tb_venda = pd.read_excel(self.filepath, sheet_name='tb_venda')
        return self.tb_loja, self.tb_sku, self.tb_ciclo, self.tb_venda

    def merge_data(self):
        """Unifica as informações de lojas, produtos e ciclos na tabela de vendas."""
        print("Realizando os merges...")
        df = self.tb_venda.merge(self.tb_loja, on='cod_loja', how='left')
        df = df.merge(self.tb_sku, on='cod_sku', how='left')
        df = df.merge(self.tb_ciclo, on='cod_ciclo', how='left')
        return df

    def clean_data(self, df):
        """Ajusta os tipos de dados e limpa eventuais inconsistências."""
        print("Limpando e tipando os dados...")
        df['qt_venda'] = df['qt_venda'].fillna(0)
        
        # Otimização de tipos para economizar memória e acelerar o modelo
        df['cod_loja'] = df['cod_loja'].astype('category')
        df['cod_sku'] = df['cod_sku'].astype('category')
        df['tp_perfil'] = df['tp_perfil'].astype('category')
        df['des_regiao'] = df['des_regiao'].astype('category')
        df['des_categoria'] = df['des_categoria'].astype('category')
        
        return df
    
    def run_pipeline(self, output_path='../data/processed_data/df_venda_final.csv'):
        """Executa a pipeline completa e salva o CSV unificado."""
        self.load_data()
        df_final = self.merge_data()
        df_final = self.clean_data(df_final)
        
        # Garante que o diretório existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"Base processada e salva em: {output_path}")
        return df_final

# Para testar no terminal:
if __name__ == "__main__":
    prep = DataPrep()
    df = prep.run_pipeline()