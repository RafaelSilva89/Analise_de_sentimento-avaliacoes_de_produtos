"""Limpeza dos textos usada no notebook, no modelo salvo e no app Streamlit."""
import html
import re

import nltk
import pandas as pd

try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords

# Negações NÃO são removidas: "not good" precisa continuar diferente de "good"
NEGACOES = {'no', 'nor', 'not'}

STOPWORDS = set(stopwords.words('english')) - NEGACOES


def limpar_texto(texto):
    """Deixa o texto em minúsculas, transforma "n't" em "not", remove pontuação e stopwords."""
    texto = html.unescape(str(texto)).lower().replace('’', "'")  # &#34; -> " (aspas que vieram como código HTML)
    texto = re.sub(r"n't\b", ' not', texto)       # don't -> do not | isn't -> is not
    palavras = re.findall(r'[a-z0-9]+', texto)     # mantém só letras e números (remove pontuação e emojis)
    return ' '.join(p for p in palavras if p not in STOPWORDS)


def contribuicao_palavras(pipeline, dados, n=8):
    """Palavras do texto que mais pesaram na previsão de um pipeline com Regressão Logística.

    Valor positivo empurra para "negativo"; valor negativo empurra para "positivo".
    """
    prep = pipeline.named_steps['prep']
    modelo = pipeline.named_steps['modelo']

    x = prep.transform(dados)
    contribuicao = pd.Series(x.multiply(modelo.coef_[0]).toarray()[0], index=prep.get_feature_names_out())

    contribuicao = contribuicao[(contribuicao != 0) & contribuicao.index.str.startswith('texto__')]
    contribuicao.index = contribuicao.index.str.replace('texto__', '')
    return contribuicao.sort_values(key=abs, ascending=False).head(n)
