
import pandas as pd
import os

class FeatureEngineering:
    def __init__(self, input_filepath='../data/processed_data/df_venda_final.csv'):
        self.input_filepath = input_filepath
        
    def get_campaign_mapping(self, df):
        '''Descobre a regra fixa de destaque na campanha.'''
        print("Extraindo regra de campanha do histórico...")
        df_camp = df[(df['flg_campanha_ciclo'] == 1) & (df['flg_campanha'] == 1)]
        
        mapping = df_camp[['tp_perfil', 'des_categoria']].drop_duplicates()
        mapping['flg_campanha'] = 1 
        return mapping

    def create_forecast_grid(self, df, next_cod_ciclo, next_nr_ordem):
        '''Cria o grid cartesiano vazio de forma dinâmica para o próximo ciclo.'''
        print(f"Criando grid de previsão para o ciclo {next_cod_ciclo}...")
        lojas = df[['cod_loja', 'tp_perfil', 'des_regiao', 'qt_espaco_gondola']].drop_duplicates()
        skus = df[['cod_sku', 'des_categoria', 'qt_espaco', 'vlr_preco_tabela', 'flg_lancamento']].drop_duplicates()
        
        grid = lojas.merge(skus, how='cross')
        grid['cod_ciclo'] = next_cod_ciclo
        grid['nr_ordem'] = next_nr_ordem
        # Assumindo a continuidade de campanha (regra do case para 202609)
        grid['flg_campanha_ciclo'] = 1 
        
        return grid
        
    def apply_campaign_rule(self, grid, mapping):
        '''Aplica a regra de destaque no grid.'''
        print("Aplicando regra de destaque no cenário futuro...")
        grid = grid.merge(mapping, on=['tp_perfil', 'des_categoria'], how='left')
        grid['flg_campanha'] = grid['flg_campanha'].fillna(0).astype(int)
        return grid
        
    def generate_lag_features(self, df):
        '''Gera features temporais (lags e media móvel) no histórico.'''
        print("Gerando features temporais no histórico...")
        df = df.sort_values(by=['cod_loja', 'cod_sku', 'nr_ordem'])
        
        df['lag_1'] = df.groupby(['cod_loja', 'cod_sku'])['qt_venda'].shift(1)
        df['lag_2'] = df.groupby(['cod_loja', 'cod_sku'])['qt_venda'].shift(2)
        df['lag_3'] = df.groupby(['cod_loja', 'cod_sku'])['qt_venda'].shift(3)
        
        df['rolling_mean_3'] = df[['lag_1', 'lag_2', 'lag_3']].mean(axis=1)
        df['diff_1'] = df['lag_1'] - df['lag_2']
        df['rolling_std_3'] = df[['lag_1', 'lag_2', 'lag_3']].std(axis=1)
        
        cols_lags = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'diff_1', 'rolling_std_3']
        df[cols_lags] = df[cols_lags].fillna(0)
        
        return df
        
    def build_features(self):
        df = pd.read_csv(self.input_filepath)
        
        # --- Cálculo Dinâmico do Próximo Ciclo ---
        max_nr_ordem = df['nr_ordem'].max()
        next_nr_ordem = max_nr_ordem + 1
        
        max_cod_ciclo = df['cod_ciclo'].max()
        next_cod_ciclo = max_cod_ciclo + 1
        # -----------------------------------------
        
        camp_mapping = self.get_campaign_mapping(df)
        
        grid_next = self.create_forecast_grid(df, next_cod_ciclo, next_nr_ordem)
        grid_next = self.apply_campaign_rule(grid_next, camp_mapping)
        
        df_hist_features = self.generate_lag_features(df)
        
        print(f"Transferindo status temporal do ciclo {max_nr_ordem} para o cenário de {next_nr_ordem}...")
        
        last_status = df_hist_features[df_hist_features['nr_ordem'] == max_nr_ordem][
            ['cod_loja', 'cod_sku', 'qt_venda', 'lag_1', 'lag_2']
        ].copy()
        
        last_status = last_status.rename(columns={
            'qt_venda': 'lag_1',
            'lag_1': 'lag_2',
            'lag_2': 'lag_3'
        })
        
        last_status['rolling_mean_3'] = last_status[['lag_1', 'lag_2', 'lag_3']].mean(axis=1)
        last_status['diff_1'] = last_status['lag_1'] - last_status['lag_2']
        last_status['rolling_std_3'] = last_status[['lag_1', 'lag_2', 'lag_3']].std(axis=1)
        
        grid_next = grid_next.merge(last_status, on=['cod_loja', 'cod_sku'], how='left')
        cols_lags = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'diff_1', 'rolling_std_3']
        grid_next[cols_lags] = grid_next[cols_lags].fillna(0)
        
        train_path = '../data/processed_data/df_train.csv'
        test_path = f'../data/processed_data/df_test_{next_cod_ciclo}.csv'
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        df_hist_features.to_csv(train_path, index=False)
        grid_next.to_csv(test_path, index=False)
        
        print(f"Bases prontas e salvas:\nTreino: {train_path}\nPrevisão: {test_path}")
        return df_hist_features, grid_next

if __name__ == '__main__':
    fe = FeatureEngineering()
    df_train, df_test = fe.build_features()
