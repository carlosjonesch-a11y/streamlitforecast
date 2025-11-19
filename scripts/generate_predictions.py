import pandas as pd
import numpy as np
import os
import sys
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Caminhos
    input_file = os.path.join('Arquivos de dados', 'mesas_preparadas.csv')
    output_dir = 'precomputed'
    output_file = os.path.join(output_dir, 'predictions.csv')

    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_file):
        logging.warning(f"Arquivo de entrada não encontrado: {input_file}. Encerrando sem erro.")
        sys.exit(0)

    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Carregar dados
        logging.info("Carregando dados...")
        df = pd.read_csv(input_file)
        df['data'] = pd.to_datetime(df['data'])
        
        # Validar colunas
        required_cols = ['data', 'demanda', 'local', 'sub_local', 'local_terciario']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Colunas faltando. Esperado: {required_cols}")
            sys.exit(1)

        # Gerar previsões para cada combinação de local
        results = []
        horizonte = 7 # dias

        groups = df.groupby(['local', 'sub_local', 'local_terciario'])
        
        logging.info(f"Gerando previsões para {len(groups)} grupos...")

        for (local, sub, tert), group in groups:
            group = group.sort_values('data').set_index('data')
            ts = group['demanda']
            
            if len(ts) < 10:
                continue

            try:
                # Modelo simples Holt-Winters
                model = ExponentialSmoothing(ts, trend='add', seasonal=None).fit()
                forecast = model.forecast(horizonte)
                
                # Criar dataframe de previsão
                future_dates = pd.date_range(start=ts.index.max() + pd.Timedelta(days=1), periods=horizonte)
                
                for date, val in zip(future_dates, forecast):
                    results.append({
                        'data': date,
                        'local': local,
                        'sub_local': sub,
                        'local_terciario': tert,
                        'previsao': max(0, val), # Garantir não negativo
                        'modelo': 'Holt-Winters'
                    })
            except Exception as e:
                logging.warning(f"Erro ao processar grupo {local}/{sub}/{tert}: {e}")
                continue

        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False)
            logging.info(f"Previsões salvas em {output_file}")
        else:
            logging.warning("Nenhuma previsão gerada.")

    except Exception as e:
        logging.error(f"Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
