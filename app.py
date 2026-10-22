import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.texto import contribuicao_palavras

st.set_page_config(page_title="Análise de Sentimento 💬", page_icon="💬")
st.title("💬 Análise de Sentimento de Reviews")
st.markdown("Escreva um review **em inglês** de um produto Amazon Alexa e descubra se ele é **positivo** ou **negativo**.")

CAMINHO_MODELO = Path(__file__).parent / "Model" / "modelo_sentimento.pkl"


### Carregamento do modelo
@st.cache_resource
def carregar_modelo(caminho):
    with open(caminho, "rb") as arquivo:
        return pickle.load(arquivo)


if not CAMINHO_MODELO.exists():
    st.error("Modelo não encontrado. Execute o notebook até a Fase 05 para gerar `Model/modelo_sentimento.pkl`.")
    st.stop()

modelo = carregar_modelo(CAMINHO_MODELO)
pipeline, limiar = modelo["pipeline"], modelo["limiar"]

### Entrada do usuário
texto = st.text_area("Review", "It stopped working after two weeks and customer support did not help.", height=120)
variacao = st.selectbox("Variação do produto", modelo["variacoes"])

### Previsão
if st.button("Analisar sentimento", type="primary"):
    if not texto.strip():
        st.warning("Escreva um review para analisar.")
        st.stop()

    dados = pd.DataFrame({"verified_reviews": [texto], "variation": [variacao]})
    prob_negativo = pipeline.predict_proba(dados)[0, 1]

    if prob_negativo >= limiar:
        st.error(f"😞 **Review NEGATIVO** (score de negatividade: {prob_negativo:.0%})")
        st.markdown("➡️ **Ação sugerida:** encaminhar para o time de atendimento entrar em contato com o cliente.")
    else:
        st.success(f"😀 **Review POSITIVO** (score de negatividade: {prob_negativo:.0%})")

    st.caption(f"O review é classificado como negativo quando o score passa de {limiar:.0%} "
               "(limiar escolhido para encontrar a maioria dos clientes insatisfeitos).")

    # Palavras que mais pesaram na decisão
    palavras = contribuicao_palavras(pipeline, dados)
    if not palavras.empty:
        st.markdown("#### Palavras que mais pesaram")
        tabela = pd.DataFrame({
            "Palavra": palavras.index,
            "Empurra para": ["Negativo 😞" if v > 0 else "Positivo 😀" for v in palavras.values],
            "Peso": palavras.values.round(2),
        })
        st.dataframe(tabela, hide_index=True, width="stretch")
