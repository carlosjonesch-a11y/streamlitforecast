# Guia rápido: inicializar repo Git, commitar e enviar para Heroku

Atenção: você já configurou `git config --global user.name` e `git config --global user.email`, então commits serão assinados com esses dados.

## 1) Inicializar o repositório (criar o .git local)
Abra o PowerShell no diretório do projeto (ex.: `G:\Meu Drive\Vscode\Streamlit`) e rode:

```powershell
# entrar na pasta do projeto
Set-Location "G:/Meu Drive/Vscode/Streamlit"

# inicializar o repositório
git init

# conferir status (arquivos não versionados)
git status
```

## 2) Adicionar e commitar arquivos (por ex. Procfile, runtime, requirements)

```powershell
# Adiciona todos os arquivos (alternativa: listar só os que deseja)
git add .

git commit -m "Initial commit: add Procfile, runtime, requirements and README"
```

## 3) Verificar autor do commit

```powershell
# mostra o autor/autor-email do último commit
git show -s --format='%an <%ae>' HEAD
```

## 4) Configurar remote e enviar ao Heroku
Se quiser usar Heroku (presume que tem Heroku CLI):

```powershell
# fazer login Heroku (será aberta uma janela do navegador)
heroku login

# criar app (se ainda não tiver) — substitua nome-do-app
heroku create nome-do-app

# enviar branch main (ou master)
git push heroku main

# acompanhar logs
heroku logs --tail
```

## 5) Caso queira também enviar para GitHub
1. Crie o repositório no GitHub.
2. No seu computador configure o 'origin' como remote do GitHub:
```powershell
# adicionar o remote origin do GitHub
git remote add origin https://github.com/usuario/nome-do-repo.git

# push branch main para origin
git push -u origin main
```

## 6) Corrigir autor do último commit (se necessário)
Se você acabou de commitar e o autor está errado, corrija assim:

```powershell
# reescreve o último commit sem mudar a mensagem
git commit --amend --author="Carlos <carlos_tp4@hotmail.com>" --no-edit

# se já havia enviado para o remoto, force push (cuidado com reescrita de histórico)
git push --force origin main
```

## 7) Verificar se está em repo (comando que você usou)

```powershell
# se retornar `true` você está dentro de um repo
git rev-parse --is-inside-work-tree
```

Se `fatal: not a git repository ...` significa que você precisa executar `git init` na pasta projeto.

---

Se quiser, eu posso também:
- Gerar um workflow `GitHub Actions` para deploy automático no Heroku.
- Ajudar a corrigir autor em commits antigos (se já tiver commits em outro e-mail).
- Incluir instruções de como testar `prophet` no Heroku caso haja erro de build.

Quer que eu adicione o `GIT-SETUP.md` ao README principal também ou basta deixar como arquivo separado?