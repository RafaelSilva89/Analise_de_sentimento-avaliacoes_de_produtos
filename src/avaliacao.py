"""Funções de avaliação de classificadores: métricas completas, relatório de validação, limiar, custo e ganho."""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from src.graficos import salvar

ROTULOS = ['Positivo (0)', 'Negativo (1)']


def calcular_metricas_completas(y_true, y_pred, y_proba):
    """Todas as métricas de um classificador binário (classe de interesse = 1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, zero_division=0),
        'Recall (Sensibilidade)': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
        'Especificidade': tn / (tn + fp) if (tn + fp) else 0,
        'Valor Preditivo Negativo': tn / (tn + fn) if (tn + fn) else 0,
        'Verdadeiros Positivos': tp, 'Verdadeiros Negativos': tn,
        'Falsos Positivos': fp, 'Falsos Negativos': fn,
    }


def relatorio_validacao(nome, modelo, X_treino, y_treino, X_val, y_val, limiar=0.5, nome_arquivo=None):
    """Avalia um modelo já treinado na validação, no formato padrão do projeto.

    Mostra: amostra gabarito x probabilidade x previsão, métricas, AUC no treino x validação (sobreajuste),
    relatório de classificação e matriz de confusão. Devolve (métricas, probabilidades da validação).
    X_treino/y_treino devem ser os dados de treino ORIGINAIS (sem cópias do balanceamento).
    """
    proba = modelo.predict_proba(X_val)[:, 1]
    previsto = (proba >= limiar).astype(int)

    print('AMOSTRA: GABARITO x PROBABILIDADE x PREVISÃO (limiar = {:.2f})'.format(limiar))
    display(pd.DataFrame({'Gabarito': np.asarray(y_val), 'Probabilidade': proba.round(3),
                          'Previsão da máquina': previsto}).head(8))

    metricas = calcular_metricas_completas(y_val, previsto, proba)
    print('=' * 50)
    print(f'MÉTRICAS DE VALIDAÇÃO: {nome}')
    print('=' * 50)
    for chave in ['Acurácia', 'Precisão', 'Recall (Sensibilidade)', 'F1-Score', 'AUC-ROC', 'PR-AUC']:
        print(f'{chave:<24} {metricas[chave]:.4f}')
    print(f'{"":<24} (sorteio: PR-AUC = {np.mean(y_val):.4f}, AUC-ROC = 0,50)')

    auc_treino = roc_auc_score(y_treino, modelo.predict_proba(X_treino)[:, 1])
    print(f'\nAUC-ROC no treino: {auc_treino:.4f} | na validação: {metricas["AUC-ROC"]:.4f}'
          f' | diferença: {auc_treino - metricas["AUC-ROC"]:+.4f}')
    metricas['AUC-ROC treino'] = auc_treino

    print('\n' + classification_report(y_val, previsto, target_names=ROTULOS, digits=4, zero_division=0))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_val, previsto, display_labels=['Positivo', 'Negativo'],
                                            cmap='Greens', colorbar=False, ax=ax)
    ax.set(xlabel='Previsto pelo modelo', ylabel='Real')
    ax.set_title(f'Matriz de confusão: {nome}', fontsize=12, loc='left')
    if nome_arquivo:
        salvar(nome_arquivo)
    plt.show()
    return metricas, proba


def plot_comparacao_modelos(metricas, taxa_positivos, nome_arquivo):
    """Painel com uma barra por modelo em cada métrica e a linha de referência de um "modelo que chuta"."""
    modelos = list(metricas)
    cores = sns.color_palette('Set2', len(modelos))
    paineis = [('Acurácia', 1 - taxa_positivos, "sempre 'positivo'"), ('Precisão', taxa_positivos, 'sorteio'),
               ('Recall (Sensibilidade)', None, None), ('F1-Score', None, None), ('AUC-ROC', 0.5, 'sorteio'),
               ('PR-AUC', taxa_positivos, 'sorteio')]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (metrica, referencia, rotulo) in zip(axes.ravel(), paineis):
        valores = [metricas[m][metrica] for m in modelos]
        ax.bar(modelos, valores, color=cores, edgecolor='black', alpha=0.85)
        for i, v in enumerate(valores):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
        if referencia is not None:
            ax.axhline(referencia, color='gray', ls='--', lw=1.2, label=f'{rotulo} = {referencia:.2f}')
            ax.legend(loc='upper right', fontsize=9)
        ax.set_title(metrica, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', labelsize=9, rotation=20)
    fig.suptitle('Comparação dos modelos na validação', fontsize=16, fontweight='bold', x=0.01, ha='left')
    plt.tight_layout()
    salvar(nome_arquivo)
    plt.show()


def limiar_otimo_f1(y, proba):
    """Limiar que maximiza o F1 (o 0,5 padrão raramente é o melhor corte operacional)."""
    precisao, revocacao, limiares = precision_recall_curve(y, proba)
    f1 = np.divide(2 * precisao * revocacao, precisao + revocacao, out=np.zeros_like(precisao),
                   where=(precisao + revocacao) > 0)
    i = int(np.argmax(f1[:-1]))
    return float(limiares[i]), float(f1[i])


def curva_custo(y, proba, custo_fn=5, custo_fp=1, limiares=np.linspace(0.05, 0.95, 91)):
    """Custo total para cada limiar: cada falso negativo custa `custo_fn` e cada falso positivo, `custo_fp`."""
    y = np.asarray(y)
    linhas = []
    for limiar in limiares:
        previsto = proba >= limiar
        fn = int(((~previsto) & (y == 1)).sum())
        fp = int((previsto & (y == 0)).sum())
        linhas.append({'Limiar': limiar, 'Falsos negativos': fn, 'Falsos positivos': fp,
                       'Custo': custo_fn * fn + custo_fp * fp})
    return pd.DataFrame(linhas)


def ganho_vs_sorteio(y, proba):
    """Quantos positivos são encontrados lendo os N casos de maior score, contra a média de um sorteio."""
    y = np.asarray(y)[np.argsort(-proba)]
    lidos = np.arange(1, len(y) + 1)
    return pd.DataFrame({'Lidos': lidos, 'Modelo': np.cumsum(y), 'Sorteio': lidos * y.mean()})


def salvar_ficha_modelo(caminho, **campos):
    """Grava a ficha do modelo (JSON): tudo o que é preciso saber para usá-lo corretamente."""
    ficha = {'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M'), **campos}
    with open(caminho, 'w', encoding='utf-8') as arquivo:
        json.dump(ficha, arquivo, ensure_ascii=False, indent=2, default=float)
    return ficha
