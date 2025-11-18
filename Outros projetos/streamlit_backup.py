import streamlit as st
import pandas as pd

# Título e descrição do aplicativo
st.title('Análise Interativa de Dados de Vendas')
st.write('Este aplicativo permite a análise de dados de vendas a partir de um arquivo CSV.')

# Upload do arquivo de dados
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    # Leitura dos dados
    data = pd.read_csv(uploaded_file)

    # Exibindo as primeiras linhas do dataframe
    st.subheader('Visualização dos Dados')
    st.write(data.head())

    # Resumo estatístico
    st.subheader('Resumo Estatístico')
    st.write(data.describe())

    # Gráfico de barras interativo
    st.subheader('Vendas por Categoria de Produto')

    # Seleção de categorias
    categorias = st.multiselect(
        'Selecione as categorias de produtos para visualizar:',
        options=data['Categoria'].unique(),
        default=data['Categoria'].unique()
    )

    # Filtrando os dados com base na seleção
    dados_filtrados = data[data['Categoria'].isin(categorias)]

    # Agrupando e somando as vendas por categoria
    vendas_por_categoria = dados_filtrados.groupby('Categoria')['Vendas'].sum()

    # Exibindo o gráfico de barras
    st.bar_chart(vendas_por_categoria)
