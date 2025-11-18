import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Configuração da página
st.set_page_config(page_title="Análise de Vendas", layout="wide")

# Título e descrição do aplicativo
st.title('📊 Análise Interativa de Dados de Vendas')
st.write('Este aplicativo permite a análise de dados de vendas a partir de um arquivo CSV com filtros avançados, gráficos dinâmicos e exportação de dados.')

# Upload do arquivo de dados
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    # Leitura dos dados
    data = pd.read_csv(uploaded_file)
    
    # Criar abas para melhor organização
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dados", "📈 Gráficos", "🎯 Análise", "📊 Estatísticas", "💾 Exportação"])
    
    with tab1:
        st.subheader('Visualização dos Dados')
        
        # Filtro de número de linhas a exibir
        num_linhas = st.slider('Quantas primeiras linhas exibir?', 1, len(data), 5)
        st.write(data.head(num_linhas))
        
        st.info(f"Total de registros no arquivo: {len(data)}")
    
    with tab2:
        st.subheader('📈 Gráficos Interativos')
        
        # Seleção de categorias
        col1, col2 = st.columns(2)
        with col1:
            categorias = st.multiselect(
                'Selecione as categorias de produtos para visualizar:',
                options=data['Categoria'].unique(),
                default=data['Categoria'].unique()
            )
        
        # Filtro de intervalo de vendas
        with col2:
            min_vendas, max_vendas = st.slider(
                'Intervalo de Vendas:',
                min_value=float(data['Vendas'].min()),
                max_value=float(data['Vendas'].max()),
                value=(float(data['Vendas'].min()), float(data['Vendas'].max())),
                step=100.0
            )
        
        # Filtrando os dados com base na seleção
        dados_filtrados = data[
            (data['Categoria'].isin(categorias)) & 
            (data['Vendas'] >= min_vendas) & 
            (data['Vendas'] <= max_vendas)
        ]
        
        if len(dados_filtrados) > 0:
            # Gráfico de barras: Vendas por categoria
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('📊 Vendas por Categoria')
                vendas_por_categoria = dados_filtrados.groupby('Categoria')['Vendas'].sum().sort_values(ascending=False)
                fig_barras = go.Figure(data=[go.Bar(x=vendas_por_categoria.index, y=vendas_por_categoria.values, marker_color='steelblue')])
                fig_barras.update_layout(xaxis_title="Categoria", yaxis_title="Vendas (R$)", height=400)
                st.plotly_chart(fig_barras, use_container_width=True)
            
            with col2:
                st.subheader('🥧 Distribuição de Vendas por Categoria')
                fig_pizza = go.Figure(data=[go.Pie(labels=vendas_por_categoria.index, values=vendas_por_categoria.values)])
                fig_pizza.update_layout(height=400)
                st.plotly_chart(fig_pizza, use_container_width=True)
            
            # Gráfico de Top 10 Produtos
            st.subheader('🏆 Top 10 Produtos Mais Vendidos')
            top_produtos = dados_filtrados.nlargest(10, 'Vendas')
            fig_top = go.Figure(data=[go.Bar(
                x=top_produtos['Vendas'].values,
                y=top_produtos['Produto'].values,
                orientation='h',
                marker_color='lightseagreen'
            )])
            fig_top.update_layout(xaxis_title="Vendas (R$)", yaxis_title="Produto", height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.warning('Nenhum dado corresponde aos filtros selecionados.')
    
    with tab3:
        st.subheader('🎯 Análise Detalhada')
        
        # Seleção de categorias para análise
        categorias_analise = st.multiselect(
            'Selecione categorias para análise:',
            options=data['Categoria'].unique(),
            default=data['Categoria'].unique(),
            key='analise_categorias'
        )
        
        dados_analise = data[data['Categoria'].isin(categorias_analise)]
        
        if len(dados_analise) > 0:
            # Distribuição de vendas por categoria (gráfico de linha)
            st.subheader('📈 Tendência de Vendas por Categoria')
            vendas_sumario = dados_analise.groupby('Categoria')['Vendas'].sum().reset_index()
            fig_linha = px.line(vendas_sumario, x='Categoria', y='Vendas', markers=True, title='Total de Vendas por Categoria')
            st.plotly_chart(fig_linha, use_container_width=True)
            
            # Scatter plot: Produtos vs Vendas colorido por categoria
            st.subheader('🎨 Análise de Produtos por Categoria')
            fig_scatter = px.scatter(dados_analise, x='Produto', y='Vendas', color='Categoria', 
                                   size='Vendas', hover_data=['Categoria', 'Vendas'],
                                   title='Distribuição de Vendas por Produto e Categoria')
            fig_scatter.update_layout(height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Tabela resumida por categoria
            st.subheader('📋 Resumo por Categoria')
            resumo_categoria = dados_analise.groupby('Categoria').agg({
                'Vendas': ['sum', 'mean', 'min', 'max', 'std'],
                'Produto': 'count'
            }).round(2)
            resumo_categoria.columns = ['Total', 'Média', 'Mínimo', 'Máximo', 'Desvio Padrão', 'Qtd Produtos']
            st.dataframe(resumo_categoria)
    
    with tab4:
        st.subheader('📊 Resumo Estatístico')
        
        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total de Vendas", f"R$ {data['Vendas'].sum():,.2f}")
        with col2:
            st.metric("📊 Média de Vendas", f"R$ {data['Vendas'].mean():,.2f}")
        with col3:
            st.metric("📦 Quantidade de Produtos", len(data))
        with col4:
            st.metric("🏷️ Categorias Únicas", data['Categoria'].nunique())
        
        # Métricas adicionais
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Maior Venda", f"R$ {data['Vendas'].max():,.2f}")
        with col2:
            st.metric("Menor Venda", f"R$ {data['Vendas'].min():,.2f}")
        with col3:
            st.metric("Desvio Padrão", f"R$ {data['Vendas'].std():,.2f}")
        
        # Tabela de estatísticas por categoria
        st.subheader('Estatísticas Detalhadas por Categoria')
        stats_categoria = data.groupby('Categoria')['Vendas'].agg(['sum', 'mean', 'count', 'min', 'max', 'std']).round(2)
        stats_categoria.columns = ['Total', 'Média', 'Quantidade', 'Mínimo', 'Máximo', 'Desvio Padrão']
        st.dataframe(stats_categoria, use_container_width=True)
        
        # Resumo completo dos dados
        st.subheader('Resumo Estatístico Completo de Vendas')
        st.write(data[['Vendas']].describe())
    
    with tab5:
        st.subheader('💾 Exportação de Dados')
        
        # Opção 1: Exportar dados filtrados
        st.write('**Opção 1: Dados com Filtros Aplicados**')
        col1, col2 = st.columns(2)
        with col1:
            categorias_export = st.multiselect(
                'Selecione categorias para exportar:',
                options=data['Categoria'].unique(),
                default=data['Categoria'].unique(),
                key='export_categorias'
            )
        
        with col2:
            min_export, max_export = st.slider(
                'Intervalo de vendas para exportar:',
                min_value=float(data['Vendas'].min()),
                max_value=float(data['Vendas'].max()),
                value=(float(data['Vendas'].min()), float(data['Vendas'].max())),
                step=100.0,
                key='export_vendas'
            )
        
        dados_export = data[
            (data['Categoria'].isin(categorias_export)) & 
            (data['Vendas'] >= min_export) & 
            (data['Vendas'] <= max_export)
        ]
        
        # Botão para exportar CSV
        csv_export = dados_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Baixar CSV Filtrado',
            data=csv_export,
            file_name='vendas_exportadas.csv',
            mime='text/csv'
        )
        
        st.success(f"✅ Total de registros para exportar: {len(dados_export)}")
        
        # Opção 2: Exportar arquivo original
        st.write('**Opção 2: Arquivo Original Completo**')
        csv_original = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Baixar CSV Original',
            data=csv_original,
            file_name='vendas_original.csv',
            mime='text/csv'
        )
else:
    st.info('👈 Faça upload de um arquivo CSV para começar a análise.')
