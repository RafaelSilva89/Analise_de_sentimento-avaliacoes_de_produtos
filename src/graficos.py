"""Funções de gráficos usadas no notebook. Cada gráfico é salvo na pasta Imagens/ do projeto."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

PASTA_IMAGENS = Path(__file__).resolve().parents[1] / 'Imagens'

COR_POSITIVO = '#4C9F70'
COR_NEGATIVO = '#D1495B'


def salvar(nome_arquivo):
    plt.savefig(PASTA_IMAGENS / f'{nome_arquivo}.png', bbox_inches='tight', dpi=110)


def titulo(texto, subtitulo=None):
    plt.suptitle(texto, fontsize=16, x=0.01, ha='left', y=1.02)
    if subtitulo:
        plt.figtext(0.01, 0.95, subtitulo, ha='left', fontsize=11, color='#656565', style='italic')


def plot_distribuicao_alvo(serie, titulo_grafico, nome_arquivo, rotulos=('Positivo', 'Negativo')):
    """Barras com quantidade e % de cada classe (0 = positivo, 1 = negativo)."""
    contagem = serie.value_counts().sort_index()
    percentual = serie.value_counts(normalize=True).sort_index() * 100

    plt.figure(figsize=(8, 4.5))
    ax = plt.bar(list(rotulos), contagem.values, color=[COR_POSITIVO, COR_NEGATIVO])
    for barra, qtd, pct in zip(ax, contagem.values, percentual.values):
        plt.text(barra.get_x() + barra.get_width() / 2, qtd, f'{qtd:,} ({pct:.1f}%)'.replace(',', '.'),
                 ha='center', va='bottom', fontsize=11)
    plt.ylabel('Quantidade de reviews')
    sns.despine(left=True)
    titulo(titulo_grafico)
    salvar(nome_arquivo)
    plt.show()


def plot_barras(serie, titulo_grafico, nome_arquivo, xlabel='', ylabel='', cor=COR_POSITIVO, horizontal=False,
                formato='{:.0f}', subtitulo=None):
    """Barras com o valor escrito em cada barra."""
    plt.figure(figsize=(10, 5) if not horizontal else (10, max(4, len(serie) * 0.35)))
    if horizontal:
        ax = sns.barplot(x=serie.values, y=serie.index.astype(str), color=cor)
    else:
        ax = sns.barplot(x=serie.index.astype(str), y=serie.values, color=cor)
    ax.bar_label(ax.containers[0], labels=[formato.format(v) for v in serie.values], padding=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not horizontal:
        plt.xticks(rotation=45 if len(serie) > 6 else 0, ha='right' if len(serie) > 6 else 'center')
    sns.despine(left=True)
    titulo(titulo_grafico, subtitulo)
    salvar(nome_arquivo)
    plt.show()


def plot_nuvem(texto, titulo_grafico, nome_arquivo, cor='Greens'):
    """Nuvem de palavras: quanto maior a palavra, mais ela aparece."""
    nuvem = WordCloud(width=1200, height=600, background_color='white', colormap=cor,
                      random_state=42).generate(texto)
    plt.figure(figsize=(12, 6))
    plt.imshow(nuvem, interpolation='bilinear')
    plt.axis('off')
    titulo(titulo_grafico)
    salvar(nome_arquivo)
    plt.show()


def plot_top_palavras(positivas, negativas, titulo_grafico, nome_arquivo, xlabel='Frequência'):
    """Duas colunas de barras horizontais: palavras dos reviews positivos x negativos."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=positivas.values, y=positivas.index, color=COR_POSITIVO, ax=ax[0])
    ax[0].set(title='Reviews positivos', xlabel=xlabel, ylabel='')
    sns.barplot(x=negativas.values, y=negativas.index, color=COR_NEGATIVO, ax=ax[1])
    ax[1].set(title='Reviews negativos', xlabel=xlabel, ylabel='')
    sns.despine(left=True)
    titulo(titulo_grafico)
    plt.tight_layout()
    salvar(nome_arquivo)
    plt.show()
