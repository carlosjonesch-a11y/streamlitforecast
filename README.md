# streamlitforecast
**Projeto Streamlit - Sistema de Previsão de Séries Temporais**

Plataforma integrada para previsão de demanda com 6 modelos de séries temporais: Prophet, TBATS, CatBoost, ARIMA, SARIMA e Holt-Winters.

**Compatibilidade**:
- **Python:** 3.14
- **Dependências principais:** Streamlit, Pandas, Plotly, Prophet, TBATS, CatBoost, Statsmodels, Scikit-learn

**Requisitos (recomendado)**:
- Python 3.14
- Visual C++ Build Tools (apenas se for compilar extensões nativas localmente)

---

## ⚡ Execução Rápida (Recomendado)

### Windows CMD (`.bat`):
```cmd
run.bat
```

### Windows PowerShell (`.ps1`):
```powershell
.\run.ps1
```

Ambos os scripts:
- ✅ Ativam o ambiente virtual automaticamente
- ✅ Instalam/atualizam dependências
- ✅ Iniciam o app em `http://localhost:8501`

---

## 🎯 Funcionalidades

O aplicativo oferece uma plataforma completa para previsão de séries temporais com:

### 6 Modelos Inclusos:
1. **Prophet** - Modelo Facebook com sazonalidade automática
2. **TBATS** - Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components
3. **CatBoost** - Gradient Boosting com engenharia de features (lags e médias móveis)
4. **ARIMA** - AutoRegressive Integrated Moving Average
5. **SARIMA** - Seasonal ARIMA com padrões sazonais
6. **Holt-Winters** - Suavização exponencial com tendência e sazonalidade

### Funcionalidades:
- ✅ Upload de arquivos CSV com séries temporais
- ✅ Seleção dinâmica de localização (local, sub-local, local terciário)
- ✅ Filtro de horizonte de previsão (7 a 365 dias)
- ✅ Visualização de série temporal histórica
- ✅ Execução simultânea de 6 modelos
- ✅ Tabela consolidada com todas as previsões
- ✅ Gráficos interativos (Plotly) com comparação de modelos
- ✅ Intervalos de confiança (95%) quando disponível
- ✅ Download de previsões em CSV
- ✅ Análise individual de cada modelo

---

## 📊 Estrutura de Dados Esperada

O arquivo CSV deve conter exatamente 5 colunas:

```
data,demanda,local,sub_local,local_terciario
2024-01-01,150,São Paulo,Zona Leste,Vila Mariana
2024-01-02,160,São Paulo,Zona Leste,Vila Mariana
...
```

**Requisitos:**
- **data**: Data no formato YYYY-MM-DD (obrigatório)
- **demanda**: Valor numérico a ser previsto (obrigatório)
- **local**: Localização principal (texto)
- **sub_local**: Sub-localização (texto)
- **local_terciario**: Localização terciária (texto)

**Observações:**
- Mínimo 10 registros por localização para executar previsões
- Dados históricos com tendências claras geram melhores resultados
- Recomenda-se pelo menos 30 dias de dados históricos

---

## 📥 Exemplo de Dataset

Arquivo incluído: `dados_series_temporais.csv`

Você pode usar este arquivo para testes iniciais do sistema.

---

## 📋 Dependências (requirements.txt)

```
pyarrow==22.0.0
streamlit>=1.50.0
pandas>=2.0.0
plotly>=5.0.0
prophet>=1.1.0
tbats>=1.1.0
catboost>=1.2.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
numpy>=1.23.0
```

---

## 🚀 Como Usar

### 1. Preparar dados (CSV)
- Criar arquivo CSV com colunas: data, demanda, local, sub_local, local_terciario
- Mínimo 10 registros por localização

### 2. Abrir a aplicação
```cmd
run.bat
```
Ou no PowerShell:
```powershell
.\run.ps1
```

### 3. Fazer upload
- Clique em "Upload CSV" e selecione o arquivo

### 4. Selecionar localização
- Escolha local, sub-local e local terciário

### 5. Gerar previsões
- Defina horizonte (dias)
- Clique em "Gerar Previsões"
- Visualize resultados em abas separadas

### 6. Baixar resultados
- Tabela consolidada em CSV
- Análise individual de cada modelo

---

## 📈 Guia de Interpretação dos Modelos

- **Prophet**: Ideal para dados com padrões sazonais claros e múltiplos anos de histórico
- **TBATS**: Excelente para capturar múltiplas sazonalidades
- **CatBoost**: Bom para relações não-lineares entre lags e demanda
- **ARIMA**: Clássico para séries estacionárias ou com tendência simples
- **SARIMA**: Recomendado para dados com sazonalidade forte
- **Holt-Winters**: Simples e rápido, bom para dados com padrão estável

**Média**: Combina todas as previsões (recomendado para decisões críticas)

---

## 💡 Dicas

- Compare os modelos usando o gráfico na aba "Comparação"
- Valide previsões contra dados historicamente conhecidos
- Use a média de múltiplos modelos para maior robustez
- Dados com ruído podem beneficiar de suavização prévia
- Estude os intervalos de confiança para entender incerteza

---

## 🔧 Instalação Manual (venv)

Se preferir instalação manual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
streamlit run app.py
```

---

## 📝 Histórico de Versões

- **v2.0** (Nov 2025): Sistema completo de previsão com 6 modelos, tabelas consolidadas e gráficos interativos
- **v1.0** (Nov 2025): App inicial de análise de vendas

---

## 📦 Como gerar/update o requirements.txt (fixar dependências do seu venv)

1. Ative o seu ambiente virtual:

```powershell
# PowerShell
& ".\.venv\Scripts\Activate.ps1"
```
ou (somente para esta sessão):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& ".\.venv\Scripts\Activate.ps1"
```

2. Gere o arquivo com todas as dependências pinadas (recomendado antes do deploy):

```powershell
pip freeze > requirements.txt
```

---

## 🔐 Como proteger seu app no Streamlit Cloud (exemplo simples)

Você pode forçar o app a pedir uma senha antes de mostrar conteúdo. Aqui está uma forma simples e segura se usada com hash:

1. Gere o hash SHA256 da sua senha localmente (exemplo em Python):

```powershell
python - <<'PY'
import hashlib
print(hashlib.sha256(b"SUA_SENHA_AQUI").hexdigest())
PY
```

2. No Streamlit Cloud → App settings → Secrets, adicione uma entrada:

```
ADMIN_PW_HASH="<sha256_hex_from_step_1>"
```

3. No início do seu `app.py` incluímos uma checagem que compara a senha informada pelo usuário com o hash
armazenado. Se incorreta, o app não mostra conteúdo.

Exemplo (o app já inclui essa checagem básica; confira o início de `app.py`).

Observações:
- Isto é conveniente para impedir acesso casual, mas não substitui autenticação completa para produção.
- Se preferir autenticação de usuários (multi-usuário), veja `streamlit-authenticator` ou OAuth via Google/GitHub.

3. Commit e push para o repositório antes do deploy Heroku.

---

<!-- Heroku section removed — deploy no Streamlit Cloud é recomendado. -->

---

## ☁️ Deploy no Streamlit Cloud (recomendado para apps Streamlit)

Streamlit Cloud é a forma mais simples de publicar um app Streamlit gratuitamente. O repositório já contém um `app.py` — use esse arquivo como o ponto de entrada do app.

Passos rápidos:

1. Faça push do repositório para GitHub.
2. Acesse https://share.streamlit.io e entre com sua conta do GitHub.
3. Clique em "New app" → escolha o repositório e a branch `main`.
4. No campo "App file", defina `app.py` (ou a rota correta se seu app estiver em uma subpasta).
5. Em "Advanced settings" você pode definir a versão do Python e variáveis de ambiente; normalmente o Streamlit Cloud lê o `requirements.txt` para instalar dependências.

Segredos e senha de acesso:
- Para proteger o app com a senha `ADMIN_PW_HASH` (comodidade local já implementada), vá em App settings → Secrets e adicione:

```
ADMIN_PW_HASH="<sha256_hex_da_sua_senha>"
```

Gere o hash localmente (exemplo em Python):

```powershell
python - <<'PY'
import hashlib
print(hashlib.sha256(b"SUA_SENHA_AQUI").hexdigest())
PY
```

Observações:
- O `requirements.txt` é usado pelo Streamlit Cloud para instalar dependências — valide se `streamlit` e libs opcionais como `prophet` estão listadas (o arquivo `requirements.txt` já está pinado). Se tiver problemas com builds (Prophet/Stan), teste localmente antes.

Dica para desenvolvimento local:
- Você pode copiar o arquivo `secrets_template.toml` para `.streamlit/secrets.toml` e preencher `ADMIN_PW_HASH` com o hash gerado; o app reconhecerá o segredo localmente via `st.secrets`. O `.streamlit/secrets.toml` está no `.gitignore` por segurança.
- Se usar arquivos de dados grandes, mantenha-os fora do repositório (por exemplo, coloque no S3) — o Streamlit Cloud tem limites de espaço.

---

---

## 🛠️ Reescrever autor/committer de commits antigos (avançado)

Se você precisa corrigir autor/email de commits já enviados (por exemplo: você cometeu com e-mail errado), é possível reescrever o histórico — porém isso altera hashes dos commits: se outras pessoas já fizeram pull desse repositório, a reescrita irá criar divergências.

Recomendações antes de reescrever o histórico:
- Faça backup do repositório (clone mirror):

```bash
git clone --mirror https://github.com/usuario/seu-repo.git
cd seu-repo.git
```

- Use o `git filter-repo` (recomendado, substitui `git filter-branch`). Instale: `pip install git-filter-repo` ou use o script oficial.

Exemplo para substituir o autor antigo por um novo (no repositório `--mirror`):

```bash
git filter-repo --commit-callback '
	if commit.author_email == b"old@example.com":
		commit.author_name = b"Carlos"
		commit.author_email = b"carlos_tp4@hotmail.com"
		commit.committer_name = b"Carlos"
		commit.committer_email = b"carlos_tp4@hotmail.com"
'

# Depois, force push para o remote (cuidado):
git push --force --tags origin 'refs/heads/*'
```

Alternativa: `git filter-branch` (deprecated) ou utilitários como `BFG Repo-Cleaner`.

IMPORTANTE: Antes de forçar push, comunique a equipe e peça que façam backup de branches; quem já tem clone deverá re-clonar ou rebase os alterações locais.

---

## 🚀 Deploy automático via GitHub Actions

Se você quiser que o seu app seja implantado automaticamente sempre que fizer push no `main`, é possível usar um workflow GitHub Actions.

1. Se você quiser deploy automático para outra plataforma (Heroku, Render, etc.), adicione os secrets/credentials necessárias no repositório GitHub e escreva um workflow compatível.
2. Este template prioriza o Streamlit Cloud — se desejar outras plataformas, crie um workflow customizado e adicione os secrets necessários.

Observações:
- Você pode restringir deploy a outras branches ou adicionar etapas de build, lint e testes antes do deploy.
-- Ao usar `prophet` em qualquer provedor, verifique os logs se ocorrer falha no build (algumas plataformas exigem dependências de SO/compilação nativa).

---

## 📌 Referência Rápida de Git

Um resumo rápido de criação / commit / push está no arquivo `GIT-SETUP.md` — recomendamos centralizar as operações a partir daí para evitar confusão entre `global` e `local` configurações do Git.

Se quiser automatizar deploy via GitHub Actions, você pode customizar um workflow — nós removemos o exemplo específico para Heroku.

(versão remota — o que veio do GitHub)
