<#
Script: git_push_fallback.ps1
Purpose: Try to push branch to remote using SSH; if push fails, create a branch and upload `precompute.yml` via GitHub REST API (requires PAT).

Usage: from repository root
  pwsh .\scripts\git_push_fallback.ps1 -Branch fast-deploy-minimal-no-prophet -Repo "carlosjonesch-a11y/streamlitforecast"

Notes:
- This script does not store tokens. Use `setx GITHUB_TOKEN "ghp_..."` (or set in current session) before running.
- Running the API fallback will create a branch on GitHub and a Pull Request for you to merge.
#>

param(
    [string]$Branch = (git branch --show-current 2>$null).Trim(),
    [string]$Repo = 'carlosjonesch-a11y/streamlitforecast',
    [string]$WorkflowPath = '.github/workflows/precompute.yml'
)

function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-ErrorAndExit($m){ Write-Host "[ERROR] $m" -ForegroundColor Red; exit 1 }

# Basic checks
if (-not (Get-Command git -ErrorAction SilentlyContinue)){
    Write-ErrorAndExit 'git is not installed or not available in PATH. Install Git and re-run the script.'
}

if (-not (Test-Path ".git")){
    Write-ErrorAndExit "This script must be run inside a Git repository root. cd to the repo and run again.";
}

if (-not $Branch){
    Write-ErrorAndExit "No branch selected. Use -Branch to pass a branch name."
}

Write-Info "Current branch: $Branch"

# Try regular push
function Invoke-GitPush {
    param($branch)
    Write-Info "Attempting push for branch $branch (HTTPS/SSH as currently configured)"
    $p = git push --set-upstream origin $branch 2>&1
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { Write-Info 'Push completed successfully.'; return $true }
    Write-Host $p
    Write-Info 'Push failed.'
    return $false
}

# Try to set remote to SSH
function Switch-RemoteToSSH {
    Write-Info 'Trying to switch remote to SSH URL (git@github.com:owner/repo.git)'
    $current = git remote get-url origin 2>$null
    if ($current -match '^git@'){
        Write-Info 'Remote already uses SSH.'; return $true
    }
    # Try to derive (owner/repo) from remote
    if ($current -match 'github.com[:/](.+?)(\.git)?$'){
        $ownerRepo = $Matches[1]
        $sshUrl = "git@github.com:$ownerRepo.git"
        git remote set-url origin $sshUrl
        Write-Info "Set remote to $sshUrl"
        return $true
    }
    Write-Info 'Could not parse current remote; leaving unchanged.'
    return $false
}

# Fallback: create branch and file via GitHub API
function New-FileViaAPI {
    param($ownerRepo, $branchName, $path)

    if (-not $env:GITHUB_TOKEN){
        Write-ErrorAndExit 'GITHUB_TOKEN not set. Please set environment variable GITHUB_TOKEN with a PAT.'
    }

    $headers = @{ Authorization = "Bearer $env:GITHUB_TOKEN"; 'User-Agent' = 'PowerShell' }
    $uriRef = "https://api.github.com/repos/$ownerRepo/git/ref/heads/main"
    Write-Info "Get main ref: $uriRef"
    $mainRef = Invoke-RestMethod -Uri $uriRef -Headers $headers -Method Get
    $shaMain = $mainRef.object.sha
    Write-Info "Main SHA: $shaMain"

    # Create a new branch
    $newRef = "refs/heads/$branchName"
    $bodyNewRef = @{ ref = $newRef; sha = $shaMain } | ConvertTo-Json
    Write-Info "Creating branch $branchName"
    [void](Invoke-RestMethod -Uri "https://api.github.com/repos/$ownerRepo/git/refs" -Headers $headers -Method Post -Body $bodyNewRef -ContentType 'application/json')

    # If we can't read the workflow file locally, error
    if (-not (Test-Path $path)){
        Write-ErrorAndExit "Local path $path not found. Make sure you're running this script from the repository root."
    }

    $content = Get-Content $path -Raw
    $encoded = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($content))
    $payload = @{ message = "Add $path via API"; content = $encoded; branch = $branchName } | ConvertTo-Json

    $uriContent = "https://api.github.com/repos/$ownerRepo/contents/$(($path -replace '/','%2F'))"
    Write-Info "Uploading file to $uriContent"
    $resp = Invoke-RestMethod -Uri $uriContent -Headers $headers -Method PUT -Body $payload -ContentType 'application/json'

    Write-Info "File uploaded; create a Pull Request to merge $branchName into main."
    $prPayload = @{ title = "Add precompute workflow via fallback"; head = $branchName; base='main'; body='Automated: adding precompute.yml via fallback script'} | ConvertTo-Json
    $prResp = Invoke-RestMethod -Uri "https://api.github.com/repos/$ownerRepo/pulls" -Headers $headers -Method Post -Body $prPayload -ContentType 'application/json'
    Write-Info "PR created: $($prResp.html_url)"
}

# Main
if (Invoke-GitPush -branch $Branch){ exit 0 }

Write-Info "Push failed. Attempt SSH fallback..."
Switch-RemoteToSSH

if (Invoke-GitPush -branch $Branch) { exit 0 }

Write-Info 'Push still failed after switching to SSH.'
Write-Info 'Will try GitHub API fallback (requires GITHUB_TOKEN environment variable).' 

$ownerRepo = $Repo
$ok = $false
try{ New-FileViaAPI -ownerRepo $ownerRepo -branchName $Branch -path $WorkflowPath; $ok = $true } catch { Write-Host $_; $ok = $false }

if (-not $ok){
    Write-ErrorAndExit 'All attempts failed. Please try again later or use the GitHub web interface to push the changes/PR.'
}

Write-Info 'Done.'