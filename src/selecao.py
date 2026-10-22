"""Seleção de variáveis por dois métodos: importância do Random Forest e Informação Mútua.

Funcionam com a matriz esparsa do TF-IDF (milhares de colunas), recebendo os nomes das colunas à parte.
"""
from functools import partial

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def importancia_random_forest(X, y, nomes, semente=42):
    """Importância de cada coluna para um Random Forest (quanto ela ajudou a separar as classes)."""
    modelo = RandomForestClassifier(random_state=semente, n_jobs=-1).fit(X, y)
    return pd.Series(modelo.feature_importances_, index=nomes).sort_values(ascending=False)


def selecao_features(X, y, nomes, k, semente=42):
    """Seleção das k melhores colunas por Informação Mútua (SelectKBest + mutual_info_classif).

    A matriz do TF-IDF é convertida em presença (1 = a palavra aparece no review, 0 = não aparece). Em matriz
    esparsa, o scikit-learn trata cada coluna como discreta, e cada peso diferente do TF-IDF viraria uma
    "categoria" própria, o que distorce o cálculo. Com a presença, a pergunta fica simples: "saber se a
    palavra aparece reduz quanto a incerteza sobre o sentimento?".
    Devolve o ranking de todas as colunas (feature, score), do maior para o menor, e o seletor ajustado.
    """
    presenca = (X > 0).astype(int)
    seletor = SelectKBest(score_func=partial(mutual_info_classif, random_state=semente), k=k).fit(presenca, y)
    df_scores = (pd.DataFrame({'feature': nomes, 'score': seletor.scores_})
                 .sort_values('score', ascending=False).reset_index(drop=True))
    return df_scores, seletor
