# Registro de decisões analíticas

Este documento registra as principais decisões do projeto, no formato
**Contexto → Decisão → Alternativas consideradas → Consequência → Onde está**.
Cada decisão tem um código (DA-XX) para ser citada em commits e pull requests.

Os números vêm das saídas do notebook [`Notebooks/Análise de sentimento - avaliações de produtos - v2.ipynb`](../Notebooks/).

| Código | Decisão |
|---|---|
| [DA-01](#da-01--alvo-e-métricas) | Alvo `negativo` e métricas da classe negativa |
| [DA-02](#da-02--remoção-de-duplicados-antes-da-divisão) | Remoção de vazios e duplicados antes da divisão |
| [DA-03](#da-03--negações-mantidas-na-limpeza-do-texto) | Negações mantidas na limpeza do texto |
| [DA-04](#da-04--nota-rating-vetada-por-vazamento) | Nota (`rating`) vetada por vazamento |
| [DA-05](#da-05--divisão-602020-com-validação-cruzada-de-complemento) | Divisão 60/20/20 com validação cruzada de complemento |
| [DA-06](#da-06--regressão-logística-como-modelo-final) | Regressão Logística como modelo final |
| [DA-07](#da-07--limiar-de-0405) | Limiar de 0,405 |
| [DA-08](#da-08--score-como-ordem-de-prioridade) | Score como ordem de prioridade, não como probabilidade exata |

---

## DA-01 — Alvo e métricas

- **Contexto:** só 8,6% dos reviews são negativos. Um modelo que dissesse "todo review é positivo" acertaria 91,4%.
- **Decisão:** alvo `negativo = 1 - feedback` (a classe de interesse é a de insatisfação); métricas principais **PR-AUC**, **recall** e **F1** da classe negativa.
- **Alternativas consideradas:** acurácia, descartada porque não diferencia um modelo útil de um que nunca encontra negativos.
- **Consequência:** todos os modelos são comparados contra um sorteio (PR-AUC = 0,086; AUC-ROC = 0,50).
- **Onde está:** Fase 01 e item 2.7 da Fase 02.

## DA-02 — Remoção de duplicados antes da divisão

- **Contexto:** 715 linhas duplicadas (22,7%) e 81 textos vazios ou nulos. Uma cópia no treino e outra no teste inflaria as métricas.
- **Decisão:** remover vazios e duplicados **antes** de dividir os dados (3.150 → 2.384 reviews).
- **Alternativas consideradas:** manter os duplicados (como a versão original), descartado por vazamento entre treino e teste.
- **Consequência:** métricas mais modestas e mais honestas que as da versão original.
- **Onde está:** item 2.4 da Fase 02.

## DA-03 — Negações mantidas na limpeza do texto

- **Contexto:** a lista padrão de stopwords remove *not*, *no* e *nor*. Sem elas, *"it does not work"* e *"it works"* ficam quase iguais.
- **Decisão:** remover as stopwords **exceto as negações** (`limpar_texto` em `src/texto.py`).
- **Alternativas consideradas:** stopwords completas (versão original).
- **Consequência:** *not* é a variável mais relevante na seleção (Informação Mútua 0,223) e a de maior peso no modelo; um review com negação tem odds ratio de 6,9 de ser negativo (H5).
- **Onde está:** item 4 da Fase 02, H5 e seleção de variáveis (item 4.3 da Fase 03).

## DA-04 — Nota (`rating`) vetada por vazamento

- **Contexto:** o rótulo foi criado a partir da nota (1-2 = negativo). O objetivo é classificar textos **sem nota**.
- **Decisão:** `rating` e `feedback` ficam fora do modelo.
- **Alternativas consideradas:** usar a nota como variável, o que daria métricas perfeitas e um modelo inútil.
- **Consequência:** o teste anti-vazamento mostra precisão, recall, F1 e PR-AUC de 1,0 com a nota, contra PR-AUC de 0,647 do modelo legítimo.
- **Onde está:** "Remoção de variáveis com vazamento" (Fase 02) e item 9 da Fase 03.

## DA-05 — Divisão 60/20/20 com validação cruzada de complemento

- **Contexto:** o teste precisa ser usado uma única vez, mas a validação tem só 41 negativos (cada review muda o recall em 2,4 p.p.).
- **Decisão:** treino 60% (1.430), validação 20% (477) e teste 20% (477), estratificados; validação cruzada de 5 partes em treino + validação como complemento (média ± desvio).
- **Alternativas consideradas:** só validação cruzada (versão anterior da v2), ou só a divisão 60/20/20.
- **Consequência:** o Naive Bayes liderou na validação (PR-AUC 0,640), mas a validação cruzada mostrou que a Regressão Logística é melhor e mais estável (0,569 ± 0,038 contra 0,550 ± 0,078).
- **Onde está:** itens 4.1, 5 e 6 da Fase 03.

## DA-06 — Regressão Logística como modelo final

- **Contexto:** foram comparados XGBoost, Random Forest, LightGBM, Regressão Logística e Naive Bayes.
- **Decisão:** Regressão Logística com `class_weight='balanced'`, ajustada por Random Search (`C ≈ 8,5`, só palavras isoladas, `min_df = 1`), treinada em treino + validação.
- **Alternativas consideradas:** Naive Bayes (melhor na validação, instável na validação cruzada) e modelos de árvore (sobreajuste: AUC de 1,000 no treino).
- **Consequência:** PR-AUC de 0,595 e AUC-ROC de 0,924 no teste; pesos por palavra interpretáveis, usados pelo app para explicar cada decisão.
- **Onde está:** itens 5 a 7 da Fase 03.

## DA-07 — Limiar de 0,405

- **Contexto:** para o atendimento, perder um cliente insatisfeito custa mais do que ler um review a mais.
- **Decisão:** limiar de **0,405**, escolhido na validação, onde o F1 máximo e a meta de recall de 75% coincidem.
- **Alternativas consideradas:** 0,50 (padrão; recall de 56%) e 0,25 (custo mínimo com falso negativo = 5 × falso positivo; recall de 95%, precisão de 41,5%).
- **Consequência:** no teste, encontra 27 dos 41 negativos lendo 45 dos 477 reviews (um sorteio encontraria ~4).
- **Onde está:** itens 8.2 e 8.3 da Fase 03; `limiar_recomendado` em `Model/modelo_sentimento.json`.

## DA-08 — Score como ordem de prioridade

- **Contexto:** com `class_weight`, o modelo superestima a probabilidade de negativo (faixa de 61% previsto tem 50% de negativos reais).
- **Decisão:** manter o modelo sem recalibração e apresentar o valor como **"score de negatividade"**.
- **Alternativas consideradas:** calibrar com `CalibratedClassifierCV`, o que exigiria refazer o limiar e esconderia os pesos por palavra.
- **Consequência:** o score é usado para **ordenar a fila de leitura** (lendo 100 reviews, o time encontra 80% dos negativos).
- **Onde está:** item 8.4 da Fase 03 e "avisos" em `Model/modelo_sentimento.json`.
