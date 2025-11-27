import pandas as pd
import numpy as np

class PreprocessData:
    ''''
    Classe para pré-processamento de dados em DataFrames do Pandas
    Até a separacao entre treino e teste.
    
    Atributos:
        df_raw: DataFrame bruto a ser processado
        is_numeric: Indica se os dados são numéricos
        is_train: Indica se o DataFrame é de treino ou teste
        excluded_columns: Lista de colunas a serem excluídas
    
    Métodos:
        trunc_values: Trunca valores acima de um percentil específico
        exclude_columns: Exclui colunas desnecessárias
        adjust_numerical_columns: Ajusta colunas numéricas para tipo float
        remove_features_with_unique_value: Remove colunas com valor único
        deal_negative_values: Remove valores negativos de colunas específicas
    '''

    def __init__(self,
                 df_raw: pd.DataFrame,
                 excluded_columns:list = [],
                 target_column:str = None, 
    ):
        self.df_raw = df_raw,
        self.target_column = target_column
        self.excluded_columns = excluded_columns

    def trunc_values(self, percentile_cut: int, columns_to_trunc:list)->pd.DataFrame:
        ''''
        Função que trunca valores que estejam acima do percentil xx - por padráo 99

        percentile_cut:
                    Valor de percentil onde dados seráo truncados

        columns_to_trunc:
                    Lista de colunas que terão valores truncados

        '''
        df = self.df_raw.copy()
        for i in columns_to_trunc:
            self.df_raw[i] = np.where(
                df[i] > df[i].quantile(percentile_cut/100),
                df[i].quantile(percentile_cut/100),
                df[i]
            )   

        return df
    

    def exclude_columns(self)->pd.DataFrame:
        '''
        Função que exclui colunas desnecessárias do dataframe

        '''
        df = self.df_raw.copy()
        df = df.drop(columns=self.excluded_columns)
        return df
    
    def adjust_numerical_columns(self, columns_to_adjust:list)->pd.DataFrame:
        '''
        Função que ajusta colunas numéricas, convertendo para tipo float

        columns_to_adjust:
                    Lista de colunas que terão valores ajustados

        '''
        df = self.df_raw.copy()
        for col in columns_to_adjust:
            df[col] = df[col].astype(float)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].round(2)
        
        return df
    
    def remove_features_with_unique_value(self)->pd.DataFrame:
        '''
        Função que remove colunas com valor único

        '''
        df = self.df_raw.copy()
        nunique = df.nunique()
        cols_to_drop = nunique[nunique == 1].index
        df = df.drop(columns=cols_to_drop)
        return df
    

    def deal_negative_values(self, columns_to_check:list)->pd.DataFrame:
        '''
        Função que remove valores negativos de colunas específicas

        columns_to_check:
                    Lista de colunas que terão valores negativos removidos

        '''
        df = self.df_raw.copy()
        for col in columns_to_check:
            df[col] = np.where(df[col] < 0, 0, df[col])
        
        return df
    

    def split_train_test(self, test_size:float=0.2):
        '''
        Função que separa o DataFrame em conjuntos de treino e teste

        target_column:
                    Nome da coluna alvo

        test_size:
                    Proporção do conjunto de teste (padrão 0.2)

        '''
        from sklearn.model_selection import train_test_split

        df = self.df_raw.copy()
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test
    

    def save_processed_data(self, df:pd.DataFrame, file_path:str):
        '''
        Função que salva o DataFrame processado em um arquivo CSV

        df:
                    DataFrame a ser salvo

        file_path:
                    Caminho do arquivo onde o DataFrame será salvo

        '''
        df.to_csv(file_path, index=False)