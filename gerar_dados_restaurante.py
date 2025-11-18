import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Configurações da Simulação ---
DATA_INICIO = datetime(2025, 10, 1, 11)  # Começa às 11:00 do dia 01/10/2025
DIAS_A_SIMULAR = 45
MESAS_TOTAIS = 20
NOME_ARQUIVO = "dataset_mesas_restaurante.xlsx"

# Definindo a ocupação base (média de ocupação em dias e horas "normais")
OCUPACAO_BASE = {
    11: 0.25, 12: 0.45, 13: 0.55, 14: 0.35, # Pico Almoço
    15: 0.15, 16: 0.10, 17: 0.10, 18: 0.20, # Horário Calmo
    19: 0.40, 20: 0.60, 21: 0.50, 22: 0.30, # Pico Jantar
    23: 0.15 # Final
}

# Fatores de Multiplicação para Picos e Dias Mais Cheios
FATOR_PICO_DIA_SEMANA = 1.0  # Segunda a Quarta
FATOR_PICO_FIM_SEMANA = 1.6  # Quinta a Domingo (Aumento de 60%)

# --- Geração do Dataset ---
dados = []
data_atual = DATA_INICIO

for _ in range(DIAS_A_SIMULAR):
    dia_semana_num = data_atual.weekday() # 0=Segunda, 6=Domingo
    
    # 0, 1, 2 = Seg, Ter, Qua (Dias mais fracos)
    # 3, 4, 5, 6 = Qui, Sex, Sáb, Dom (Dias mais fortes)
    if dia_semana_num >= 3:
        fator_dia = FATOR_PICO_FIM_SEMANA
    else:
        fator_dia = FATOR_PICO_DIA_SEMANA
        
    for hora in range(11, 24): # 11:00 até 23:00
        # Cria um nível de ruído aleatório para simular a variação natural
        ruido = np.random.uniform(-0.10, 0.10) 
        
        # Calcula a taxa de ocupação
        taxa_ocupacao = (OCUPACAO_BASE[hora] * fator_dia) + ruido
        
        # Garante que a taxa fique entre 0 e 1 (100%)
        taxa_ocupacao = max(0, min(1, taxa_ocupacao))
        
        # Calcula o número de mesas e arredonda para o inteiro mais próximo
        mesas_ocupadas = int(round(taxa_ocupacao * MESAS_TOTAIS))
        
        # Cria a linha de dados
        data_hora = data_atual.replace(hour=hora, minute=0)
        dados.append({
            # Mantemos Data e Hora separadas por compatibilidade, mas adicionamos DataHora conjunto
            "Data": data_hora.strftime("%Y-%m-%d"),
            "Hora": data_hora.strftime("%H:%M"),
            "DataHora": data_hora,
            "Dia da Semana": data_hora.strftime("%A"),
            "Mesas Ocupadas": mesas_ocupadas
        })

    # Avança para o próximo dia, redefinindo a hora de início
    data_atual += timedelta(days=1)
    data_atual = data_atual.replace(hour=11)

# Cria o DataFrame e exporta para Excel
df = pd.DataFrame(dados)

# Renomeando o Dia da Semana para Português
dias_pt = {
    'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
    'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
}
df['Dia da Semana'] = df['Dia da Semana'].map(dias_pt)

# Garante que a coluna 'Data' seja exportada como formato de data (opcional, mas bom para o Excel)
df['Data'] = pd.to_datetime(df['Data']) 
# Garante que DataHora é dtype datetime (Excel preserva data/hora)
df['DataHora'] = pd.to_datetime(df['DataHora'])

# Exporta para XLSX
try:
    # Remover colunas Data e Hora — manter apenas DataHora + colunas relevantes
    if 'Data' in df.columns and 'Hora' in df.columns:
        df = df.drop(columns=['Data', 'Hora'])
    # Reordenar para deixar DataHora em primeiro
    cols = ['DataHora', 'Dia da Semana', 'Mesas Ocupadas']
    df = df[cols]
    df.to_excel(NOME_ARQUIVO, index=False, engine='openpyxl')
    print(f"\n✅ Sucesso! O arquivo '{NOME_ARQUIVO}' foi gerado com 540 registros.")
    print("O dataset contém a simulação dos picos de 11h-14h e 19h-21h, com maior volume de Quinta a Domingo.")
except Exception as e:
    print(f"\n❌ Erro ao exportar para Excel: {e}")
    print("Certifique-se de que a biblioteca 'openpyxl' está instalada (pip install openpyxl).")