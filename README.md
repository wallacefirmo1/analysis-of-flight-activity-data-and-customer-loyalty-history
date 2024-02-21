# Projeto de Ciência de Dados

Este projeto foi desenvolvido com o objetivo de aplicar e demonstrar minhas habilidades em Ciência de Dados. O projeto envolve a análise de dados de atividades de voo e histórico de fidelidade de clientes.

## 1.0. Competências Demonstradas

1. **SQL**: Realizei várias consultas SQL para extrair dados específicos do banco de dados SQLite. As consultas variaram desde a seleção de todas as colunas até a aplicação de condições específicas para filtrar os dados.

2. **Pandas**: Usei a biblioteca Pandas para manipular os dados extraídos. Isso incluiu a visualização das primeiras linhas dos DataFrames, a seleção de colunas específicas e a identificação e remoção de dados faltantes.

3. **Análise de Dados**: Realizei uma análise inicial dos dados para obter insights. Isso incluiu a verificação do número de linhas e colunas, a obtenção de informações gerais sobre o DataFrame e o cálculo da média de uma coluna específica.

4. **Machine Learning**: Implementei um modelo de Árvore de Decisão usando a biblioteca Scikit-learn. O modelo foi usado para classificar os dados com base em vários atributos.

## 2.0. Código

Aqui está um breve resumo do código usado neste projeto:

```python
import sqlite3
import pandas as pd
from sklearn import tree as tr

# Conexão com o banco de dados
conn = sqlite3.connect("database.db")

# Consulta SQL
consulta_atividade = """ 
    SELECT
        fa.*
    FROM flight_activity fa
    WHERE
        fa.flights_booked > 11
"""
df_atividade = pd.read_sql_query(consulta_atividade, conn)

# Manipulação de dados com Pandas
df_atividade.head()

# Análise de dados
df_atividade.info()

# Machine Learning
X_atributos = df_dados_completos.drop( columns="loyalty_card" )
y_rotulos = df_dados_completos.loc[:, "loyalty_card"]
modelo = tr.DecisionTreeClassifier(max_depth=5)
```

## 3.0. Machine Learning

Nesta seção, utilizei a biblioteca Scikit-learn para implementar um modelo de Árvore de Decisão. O modelo foi treinado com os dados completos, excluindo a coluna “loyalty_card”, que foi usada como rótulo.

```
from sklearn import tree as tr

X_atributos = df_dados_completos.drop( columns="loyalty_card" )
y_rotulos = df_dados_completos.loc[:, "loyalty_card"]

# Definição do algoritmo
modelo = tr.DecisionTreeClassifier(max_depth=5)

# Treinamento do algoritivo
modelo_treinado = modelo.fit(X_atributos, y_rotulos)
```

## 4.0. Apresentando o Resultado

```
Após o treinamento do modelo, realizei algumas previsões usando amostras aleatórias dos atributos. As previsões foram apresentadas em termos de probabilidades para cada classe de “loyalty_card”.

X_novo = X_atributos.sample()
previsao = modelo_treinado.predict_proba(X_novo)

print( "Prob - Aurora: {:.1f}% - Nova: {:.1f}% = Star: {:.1f}%".format(100*previsao[0][0], previsao[0][1], 100*previsao[0][2]))
```

## 5.0. Painel de Visualização

Finalmente, criei um painel de visualização interativo usando a biblioteca Gradio. O painel permite que os usuários ajustem os atributos e vejam as previsões do modelo em tempo real.

```
import gradio as gr
import numpy as np

def predict(*args):
    X_novo = np.array( [args] ).reshape(1, -1)
    previsao = modelo_treinado.predict_proba( X_novo )
    
    return {"Aurora": previsao[0][0], "Nova":previsao[0][1], "Star": previsao[0][2]}
    
with gr.Blocks() as demo:
    # Titulo do Painel
    gr.Markdown(""" # Propensão de Compra """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown( """ # Atributos do Cliente """)
            year                    = gr.Slider( label= "year", minimum=2017, maximum=2018, step=1, randomize=True)
            month                   = gr.Slider( label= "month", minimum=1, maximum=12, step=1, randomize=True)
            fligths_booked          = gr.Slider( label= "fligths_booked", minimum=0, maximum=21, step=1, randomize=True)
            fligths_with_companions = gr.Slider( label= "fligths_with_companions", minimum=0, maximum=11, step=1, randomize=True)
            total_flights           = gr.Slider( label= "total_flights", minimum=0, maximum=32, step=1, randomize=True)
            distance                = gr.Slider( label= "distance", minimum=0, maximum=6293, step=1, randomize=True)
            points_accumulated      = gr.Slider( label= "points_accumulated", minimum=0.00, maximum=676.5, step=0.1, randomize=True)
            salary                  = gr.Slider( label= "salary", minimum=58486.00, maximum=407228.00, step=0.1, randomize=True)
            clv                     = gr.Slider( label= "clv", minimum=2119.89, maximum=83325.38, step=0.1, randomize=True)
            
            with gr.Row():
                gr.Markdown( """# Botão de Previsão """)
                predict_btn = gr.Button( value = "Previsao")                
        with gr.Column():
            gr.Markdown( """# Coluna 2 """)
            label = gr.Label()

    # Botão de predict
    predict_btn.click(
        fn=predict,
        inputs=[
            year,
            month, 
            fligths_booked,
            fligths_with_companions,
            total_flights,
            distance,
            points_accumulated,
            salary,
            clv
            ],
        outputs=[label])
    

demo.launch(debug=True, share=False)
```

## 6.0. Conclusão

Este projeto demonstrou minha capacidade de trabalhar com SQL, Pandas e Scikit-learn para analisar dados e implementar um modelo de Machine Learning.
