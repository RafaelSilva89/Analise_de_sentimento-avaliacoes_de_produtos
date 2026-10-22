# 💬 Análise de Sentimento de Reviews de Produtos

**Um modelo que lê avaliações de clientes e encontra automaticamente os insatisfeitos.**

<div align="center">
<img src="Imagens/Capa.PNG" width="500px">
</div>

**Python · NLP · TF-IDF · Scikit-learn · Regressão Logística · Streamlit · CRISP-DM**

| 📊 Destaques | |
|---|---|
| Reviews analisados | **2.384** (após remover 766 vazios e duplicados) |
| Desafio | Apenas **8,6%** dos reviews são negativos |
| Resultado | O modelo encontra **7 em cada 10 clientes insatisfeitos**, lendo só **11%** dos reviews |
| Principal descoberta | A palavra **"not"** é o sinal mais forte de insatisfação, e a versão original a removia do texto |
| Entrega | Modelo em **Pipeline** + **app Streamlit** que classifica um review e mostra as palavras que pesaram |

---

## 📑 Sumário

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo](#2-objetivo)
3. [Base de dados](#3-base-de-dados)
4. [Estrutura do projeto](#4-estrutura-do-projeto)
5. [Como executar](#5-como-executar)
6. [Etapas do projeto (CRISP-DM)](#6-etapas-do-projeto-crisp-dm)
7. [Principais descobertas da análise exploratória](#7-principais-descobertas-da-análise-exploratória)
8. [Construção da solução](#8-construção-da-solução)
9. [Resultado do modelo](#9-resultado-do-modelo)
10. [Palavras que mais indicam insatisfação](#10-palavras-que-mais-indicam-insatisfação)
11. [App interativo (Streamlit)](#11-app-interativo-streamlit)
12. [O que mudou da v1 para a v2](#12-o-que-mudou-da-v1-para-a-v2)
13. [Insights](#13-insights)
14. [Limites](#14-limites)
15. [Próximos passos](#15-próximos-passos)

---

## 1. Contexto do problema

Empresas recebem diariamente centenas de avaliações de clientes em sites e redes sociais. **Ler tudo manualmente é inviável**, e um cliente insatisfeito sem resposta pode virar uma reclamação pública, uma devolução ou um cliente perdido.

Este projeto usa **reviews reais de clientes dos produtos Amazon Alexa** (Echo, Echo Dot, Fire TV Stick etc.) para construir um "leitor automático" que separa os reviews **positivos** dos **negativos**.

**Fonte dos dados:** [Amazon Alexa Reviews – Kaggle](https://www.kaggle.com/sid321axn/amazon-alexa-reviews)

## 2. Objetivo

Construir um modelo que receba o **texto de um review** e indique se ele é **positivo ou negativo**, priorizando **encontrar os clientes insatisfeitos**.

> **Pergunta de negócio:** *É possível identificar automaticamente os clientes insatisfeitos a partir do que eles escrevem, para que o time de atendimento aja antes que o problema cresça?*

## 3. Base de dados

| Variável | Descrição |
|---|---|
| `rating` | Nota que o cliente deu ao produto (1 a 5) |
| `date` | Data do review (maio a julho de 2018) |
| `variation` | Variação do produto (modelo e cor, ex.: *Black Dot*, *Charcoal Fabric*) |
| `verified_reviews` | **Texto** escrito pelo cliente (em inglês) |
| `feedback` | **Rótulo**: notas 1–2 = negativo · notas 3–5 = positivo |

## 4. Estrutura do projeto

```
Analise_de_sentimento-avaliacoes_de_produtos/
├── Datasets/
│   ├── amazon_alexa.tsv                       # base original (Kaggle)
│   └── amazon_alexa_limpo.csv                 # base sem vazios e duplicados (gerada no notebook)
├── Imagens/                                   # gráficos gerados pelo notebook
├── Model/
│   ├── text_classifier.pkl                    # modelo da versão original
│   └── modelo_sentimento.pkl                  # pipeline + limiar da v2 (usado pelo app)
├── Notebooks/
│   ├── Análise de sentimento - avaliações de produtos.ipynb        # versão original
│   └── Análise de sentimento - avaliações de produtos - v2.ipynb   # versão reorganizada (CRISP-DM)
├── src/
│   ├── texto.py                               # limpeza do texto e explicação da previsão
│   └── graficos.py                            # funções de gráficos
├── app.py                                     # app Streamlit
├── requirements.txt
└── README.md
```

## 5. Como executar

```bash
# 1. Clonar o repositório
git clone https://github.com/RafaelSilva89/Analise_de_sentimento-avaliacoes_de_produtos.git
cd Analise_de_sentimento-avaliacoes_de_produtos

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Abrir o notebook e executar todas as células (gera o modelo usado pelo app)
jupyter notebook "Notebooks/Análise de sentimento - avaliações de produtos - v2.ipynb"

# 4. Abrir o app
streamlit run app.py
```

## 6. Etapas do projeto (CRISP-DM)

| Fase | O que foi feito |
|---|---|
| **01. Entendimento do negócio** | Definição do problema, da métrica (encontrar os negativos) e da entrega |
| **02. Preparação dos dados** | Remoção de vazios e duplicados, análise exploratória com 6 hipóteses e limpeza do texto |
| **03. Modelagem** | Texto → números (TF-IDF), comparação de 4 modelos, ajuste fino, escolha do limiar e teste de calibração |
| **04. Avaliação** | Métricas no teste, leitura dos acertos e erros e palavras mais importantes |
| **05. Implantação** | Pipeline salvo, simulação com reviews novos e app Streamlit |

## 7. Principais descobertas da análise exploratória

**A grande maioria dos clientes está satisfeita.**

<div align="center">
<img src="Imagens/Distribuicao%20positivos%20x%20negativos.png" width="550px">
</div>

- Só **8,6%** dos reviews são negativos. Por isso, "acertar 91%" não significa nada: um modelo que dissesse que todo review é positivo já acertaria isso.

**Cliente insatisfeito escreve mais.**

<div align="center">
<img src="Imagens/Tamanho%20por%20sentimento.png" width="650px">
</div>

- O review negativo mediano tem **163 caracteres**, mais que o dobro do positivo (**75**). Quem está insatisfeito **explica o problema**.

**Negações são o sinal mais claro de insatisfação.**

<div align="center">
<img src="Imagens/Negacao%20por%20sentimento.png" width="650px">
</div>

- **63%** dos reviews negativos têm alguma negação (*not*, *no*, *don't*), contra **20%** dos positivos.

**Alguns produtos recebem mais críticas.**

<div align="center">
<img src="Imagens/%25%20negativos%20por%20variacao.png" width="750px">
</div>

- As variações **White**, **Black** e **Black Spot** têm de 13% a 16% de reviews negativos, contra menos de 3% dos modelos com acabamento em tecido.

**O que os clientes escrevem.**

<div align="center">
<img src="Imagens/Top%20palavras%20por%20sentimento.png" width="850px">
</div>

- Os positivos falam de **uso e satisfação** (*love*, *great*, *music*, *easy*); os negativos, do **aparelho e do funcionamento** (*device*, *work*, *screen*).

## 8. Construção da solução

**1️⃣ Limpeza da base**: removemos **80 reviews vazios** e **715 duplicados (23% da base)**. Duplicados são perigosos: se a cópia de um review cai no treino e a outra no teste, o modelo é avaliado num texto que já viu, e o resultado parece melhor do que é.

**2️⃣ Limpeza do texto**: tudo em minúsculas, sem pontuação e sem palavras muito comuns que não indicam sentimento (*the*, *is*, *my*). As **negações foram mantidas**: sem elas, *"it does **not** work"* e *"it works"* ficariam quase iguais.

**3️⃣ Texto → números (TF-IDF)**: o computador não lê texto, então cada review vira uma lista de números que representa as palavras que ele contém. Palavras que aparecem em todo review (como *echo*) recebem peso baixo; palavras que diferenciam um review dos outros, peso alto.

> 💡 **Analogia:** o TF-IDF funciona como um marca-texto, destacando as palavras que realmente importam em cada review.

**🔍 Seleção de variáveis**: medimos a relevância de cada uma das 3.927 colunas (palavras e variações do produto) com dois métodos, o **Random Forest** ("quais perguntas o modelo mais usou para decidir") e a **Informação Mútua** ("quanto a palavra, sozinha, conta sobre o sentimento"). Os dois apontam **"not"** como a variável mais relevante, seguida de palavras de defeito e devolução (*work*, *device*, *back*, *return*, *stopped*). A variação do produto quase não pesa.

<div align="center">
<img src="Imagens/Informacao%20mutua.png" width="650px">
</div>

**4️⃣ Comparação de modelos**: testamos 4 combinações com validação cruzada, medindo a capacidade de **encontrar os reviews negativos**:

| Modelo | Precisão | Recall | F1 |
|---|---|---|---|
| Naive Bayes + SMOTE *(abordagem da v1)* | 37,6% | 74,4% | 49,9% |
| Regressão Logística + SMOTE | 50,9% | 64,0% | 56,6% |
| **Regressão Logística (class_weight)** ✅ | 47,6% | 70,1% | **56,6%** |
| LinearSVC (class_weight) | 59,9% | 53,1% | 55,9% |

A **Regressão Logística** foi escolhida por ter o melhor F1, um recall alto e por ser **interpretável**: cada palavra tem um peso que mostra se ela empurra o review para positivo ou negativo. Depois do ajuste fino, a qualidade do modelo (PR-AUC) subiu de **0,57 para 0,64**.

**5️⃣ Escolha do limiar**: o modelo dá um **score** de 0 a 1 para cada review. O limiar é o ponto a partir do qual o review é considerado negativo.

<div align="center">
<img src="Imagens/Escolha%20do%20limiar.png" width="700px">
</div>

- Para o atendimento, **deixar um cliente insatisfeito sem resposta custa mais** do que ler um review a mais. Por isso o limiar foi reduzido de 0,50 para **0,34**, escolhido para encontrar pelo menos 75% dos negativos.
- A escolha foi feita **só com os dados de treino**, e o teste foi usado uma única vez, no final.

## 9. Resultado do modelo

Avaliação no **teste** (477 reviews que o modelo nunca viu, sendo 41 negativos):

| Métrica | Valor | Em linguagem de negócio |
|---|---|---|
| **Recall** | **70,7%** | De cada **10 clientes insatisfeitos**, o modelo encontra **7** |
| **Precisão** | **53,7%** | De cada **2 alertas**, ~**1** é realmente um review negativo |
| **F1** | **61,1%** | Equilíbrio entre os dois (era 43% na versão original) |

<div align="center">
<img src="Imagens/Matriz%20de%20confusao.png" width="380px">
</div>

- Na prática, o time lê **54 dos 477 reviews (11%)** e encontra **29 dos 41** clientes insatisfeitos.
- **8 dos 25 alarmes falsos** são reviews **nota 3**, que o rótulo considera positivos, mas que contêm críticas (ex.: *"I try use with yale lock but isn't work"*).

## 10. Palavras que mais indicam insatisfação

<div align="center">
<img src="Imagens/Palavras%20que%20mais%20influenciam%20o%20modelo.png" width="850px">
</div>

- **"not"** é a palavra mais forte de todas.
- **"back"** e **"return"** indicam **intenção de devolver o produto**: esses clientes devem ter prioridade no atendimento.
- **"siri"** aparece quando o cliente **compara com a concorrência**.

## 11. App interativo (Streamlit)

O time de atendimento pode usar o modelo **sem precisar de código**:

1. Cole o review e escolha a variação do produto.
2. Clique em **Analisar sentimento**.
3. O app mostra se o review é **positivo ou negativo**, o **score de negatividade**, a **ação sugerida** e as **palavras que mais pesaram** na decisão.

```bash
streamlit run app.py
```

> ℹ️ O app usa o arquivo `Model/modelo_sentimento.pkl`, gerado ao executar o notebook até a Fase 05.

<!-- Adicione aqui um GIF de demonstração do app: <img src="Imagens/app.gif" width="800px"> -->

## 12. O que mudou da v1 para a v2

A versão original foi mantida no repositório para mostrar a evolução. Na v2:

| Versão original (v1) | Versão 2 | Por que importa |
|---|---|---|
| Reviews duplicados (23%) mantidos | Removidos **antes** de separar treino e teste | Evita avaliar o modelo em textos que ele já viu |
| Stopwords removiam *not*, *no*, *don't* | **Negações mantidas** | *"not good"* deixa de virar *"good"* |
| Vocabulário criado com a base inteira | Tudo dentro de um **Pipeline** treinado só no treino | Sem vazamento de informação do teste |
| Destaque para o F1 da classe positiva (93%) | Métricas da **classe negativa** (a que importa) | O F1 negativo real era 43% |
| Limiar escolhido **no teste** (e com as classes invertidas) | Limiar escolhido por validação cruzada no treino | Avaliação honesta |
| Naive Bayes x Regressão Logística | 4 modelos comparados + ajuste fino + calibração | Escolha justificada |
| Anotação dizia que nota 3 era negativa | Corrigido: nota 3 é rotulada como **positiva** | Entendimento correto do rótulo |

## 13. Insights

1. **Clientes insatisfeitos escrevem o dobro**: os reviews negativos trazem detalhes do problema, o que é informação valiosa para o time de produto.
2. **Negação é o principal sinal de insatisfação**: remover *not* na limpeza do texto apaga justamente o que mais importa.
3. **Intenção de devolução é detectável**: *back* e *return* identificam clientes prestes a devolver o produto.
4. **Alguns produtos geram mais críticas**: White, Black e Black Spot têm até 6 vezes mais reviews negativos que os modelos em tecido.
5. **O modelo reduz o trabalho manual**: lendo ~11% dos reviews, o time encontra 7 em cada 10 clientes insatisfeitos.

## 14. Limites

- **Poucos exemplos negativos** (205): o teste tem só 41, então os resultados variam alguns pontos percentuais com pequenas mudanças.
- **Rótulo imperfeito**: reviews nota 3 são considerados positivos, mas muitos contêm críticas.
- **Palavras isoladas**: o modelo não entende ironia, sarcasmo nem contexto (ex.: *"disappoint"* aparece tanto em *"does not disappoint"* quanto em críticas).
- **Somente inglês** e um período curto de dados (maio a julho de 2018).

## 15. Próximos passos

1. **Revisar o rótulo** dos reviews nota 3 e considerar uma classe "neutra".
2. **Coletar mais reviews negativos**, o ganho mais rápido para o modelo.
3. **Testar modelos de linguagem** (embeddings ou BERT/DistilBERT), que entendem contexto e sarcasmo.
4. **Monitorar em produção** a precisão dos alertas e ajustar o limiar.
5. **Publicar o app** no Streamlit Community Cloud e adicionar um GIF de demonstração.

---

📫 **Contato:** [GitHub](https://github.com/RafaelSilva89) · [Portfólio](https://rafaelsilva89.github.io/portfolioProjetos/)
