# -*- coding: utf-8 -*-
"""
API de Otimização de Gôndola e Previsão de Demanda
Framework: FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import uvicorn
import os

# Instanciando a aplicação
app = FastAPI(
    title="API Trade Marketing - Otimizador de Gôndola",
    description="Endpoint para servir previsões de vendas (LightGBM) e distribuição física (PuLP).",
    version="1.0.0"
)

# --- Modelos de Dados (Pydantic) ---
class ProdutoOtimizado(BaseModel):
    cod_sku: str
    des_categoria: str
    qt_venda_projetada: int
    qt_espaco_alocado: int

class RespostaOtimizacao(BaseModel):
    cod_loja: str
    ciclo: int
    compliance_atingido: bool
    ocupacao_percentual: float
    mix_produtos: List[ProdutoOtimizado]

# --- Variáveis Globais (Cache em Memória) ---
# Na vida real, isso poderia vir de um banco de dados (PostgreSQL/Redis)
DATA_PATH = "../data/predictions/previsao_202609_full.csv"
df_previsoes = pd.DataFrame()

@app.on_event("startup")
def carregar_dados():
    """Carrega os dados em memória quando a API sobe para respostas em milissegundos."""
    global df_previsoes
    try:
        if os.path.exists(DATA_PATH):
            df_previsoes = pd.read_csv(DATA_PATH)
            print(f"[STARTUP] Base de previsões carregada com {len(df_previsoes)} registros.")
        else:
            print(f"[STARTUP] Aviso: {DATA_PATH} não encontrado. Usando mock interno.")
            # Mock de contingência para a API não quebrar se o CSV não existir
            df_previsoes = pd.DataFrame({
                "cod_loja": ["L001", "L001", "L002"],
                "cod_sku": ["SKU10", "SKU20", "SKU10"],
                "cod_ciclo": [202609, 202609, 202609],
                "des_categoria": ["Perfumaria", "Maquiagem", "Perfumaria"],
                "qt_venda_prevista": [150, 300, 120],
                "qt_espaco_alocado": [10, 20, 8] # Simulação da saída do PuLP
            })
    except Exception as e:
        print(f"[STARTUP] Erro ao carregar dados: {e}")

# --- Endpoints ---
@app.get("/", tags=["Health Check"])
def health_check():
    """Verifica se a API está no ar."""
    return {"status": "online", "message": "API de Trade Marketing operando normalmente."}

@app.get("/otimizacao/{cod_loja}", response_model=RespostaOtimizacao, tags=["Gôndola"])
def obter_gondola_otimizada(cod_loja: str, ciclo: Optional[int] = 202609):
    """
    Retorna o plano de gôndola perfeito para uma loja específica.
    Consulta os resultados do LightGBM + PuLP.
    """
    if df_previsoes.empty:
        raise HTTPException(status_code=503, detail="Base de dados não está disponível no momento.")
        
    # Filtra os dados da loja solicitada
    df_loja = df_previsoes[(df_previsoes['cod_loja'] == cod_loja) & (df_previsoes['cod_ciclo'] == ciclo)]
    
    if df_loja.empty:
        raise HTTPException(status_code=404, detail=f"Nenhum dado de otimização encontrado para a loja {cod_loja} no ciclo {ciclo}.")
    
    # Constrói o payload de resposta
    mix = []
    for _, row in df_loja.iterrows():
        mix.append(ProdutoOtimizado(
            cod_sku=row['cod_sku'],
            des_categoria=row.get('des_categoria', 'Indefinida'),
            qt_venda_projetada=int(row['qt_venda_prevista']),
            qt_espaco_alocado=int(row.get('qt_espaco_alocado', row['qt_venda_prevista'] * 0.1)) # Mock se não tiver a coluna do PuLP
        ))
        
    resposta = RespostaOtimizacao(
        cod_loja=cod_loja,
        ciclo=ciclo,
        compliance_atingido=True, # Simulação do status do otimizador linear
        ocupacao_percentual=100.0,
        mix_produtos=mix
    )
    
    return resposta

if __name__ == "__main__":
    # Para rodar localmente: python api.py
    # Ou via terminal: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run("09_api:app", host="0.0.0.0", port=8000, reload=True)
