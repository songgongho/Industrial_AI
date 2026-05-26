param(
    [string]$PythonExe = "python",
    [string]$OutputRoot = "outputs/experiments"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."

$summaryRows = @()

function Run-Experiment {
    param(
        [string]$Name,
        [int]$NCycles,
        [int]$Epochs,
        [double]$Lr,
        [double]$AnomalyProb,
        [string]$Loss = "bce",
        [double]$Gamma = 2.0,
        [string]$LrSchedule = "none"
    )

    $outDir = Join-Path $OutputRoot $Name
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    Write-Host "`n=== Running $Name ===" -ForegroundColor Cyan
    Write-Host "n_cycles=$NCycles epochs=$Epochs lr=$Lr anomaly_prob=$AnomalyProb loss=$Loss gamma=$Gamma lr_schedule=$LrSchedule" -ForegroundColor DarkCyan

    & $PythonExe "ml/train_mvp.py" `
        --n-cycles $NCycles `
        --n-points 192 `
        --batch-size 16 `
        --epochs $Epochs `
        --lr $Lr `
        --loss $Loss `
        --gamma $Gamma `
        --lr-schedule $LrSchedule `
        --anomaly-prob $AnomalyProb `
        --output-dir $outDir

    & $PythonExe "ml/predict_and_explain.py" `
        --checkpoint (Join-Path $outDir "model_mvp.ckpt") `
        --output-dir $outDir `
        --n-cycles $NCycles `
        --n-points 192 `
        --batch-size 16 `
        --seed 42

    $metricsPath = Join-Path $outDir "metrics.json"
    if (Test-Path $metricsPath) {
        $m = Get-Content $metricsPath | ConvertFrom-Json
        $f1 = [double]$m.f1
        $auc = [double]$m.roc_auc
        $thr = $m.optimal_threshold
        Write-Host ("[{0}] f1={1:N4} roc_auc={2:N4} optimal_threshold={3}" -f $Name, $f1, $auc, $thr) -ForegroundColor Green

        $summaryRows += [pscustomobject]@{
            exp_name = $Name
            f1 = $f1
            roc_auc = $auc
            optimal_threshold = $thr
            epochs = $Epochs
            lr = $Lr
            loss_type = $Loss
            gamma = $Gamma
            lr_schedule = $LrSchedule
        }
    }
    else {
        Write-Host "[$Name] metrics.json not found" -ForegroundColor Yellow
    }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

# exp01: epochs=20 (others same)
Run-Experiment -Name "exp01" -NCycles 32 -Epochs 20 -Lr 0.001 -AnomalyProb 0.15

# exp02: n_cycles=512, epochs=20
Run-Experiment -Name "exp02" -NCycles 512 -Epochs 20 -Lr 0.001 -AnomalyProb 0.15

# exp03: exp02 + lr=1e-4
Run-Experiment -Name "exp03" -NCycles 512 -Epochs 20 -Lr 0.0001 -AnomalyProb 0.15

# exp04: exp03 + anomaly_prob=0.3
Run-Experiment -Name "exp04" -NCycles 512 -Epochs 20 -Lr 0.0001 -AnomalyProb 0.3

# exp05: --loss focal --gamma 2.0 --n-cycles 512 --epochs 20 --lr 1e-3
Run-Experiment -Name "exp05" -NCycles 512 -Epochs 20 -Lr 0.001 -AnomalyProb 0.15 -Loss "focal" -Gamma 2.0

# exp06: --lr-schedule cosine --n-cycles 512 --epochs 20 --lr 1e-3
Run-Experiment -Name "exp06" -NCycles 512 -Epochs 20 -Lr 0.001 -AnomalyProb 0.15 -LrSchedule "cosine"

# exp07: --anomaly-prob 0.3 --n-cycles 512 --epochs 20 --lr 1e-4 --loss focal --gamma 2.0
Run-Experiment -Name "exp07" -NCycles 512 -Epochs 20 -Lr 0.0001 -AnomalyProb 0.3 -Loss "focal" -Gamma 2.0

$summaryPath = "outputs/experiment_summary.csv"
$summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8

Write-Host "`n=== Experiment Summary (console) ===" -ForegroundColor Magenta
$summaryRows | ForEach-Object {
    Write-Host ("{0}: f1={1:N4}, roc_auc={2:N4}, threshold={3}, loss={4}, gamma={5}, schedule={6}" -f $_.exp_name, $_.f1, $_.roc_auc, $_.optimal_threshold, $_.loss_type, $_.gamma, $_.lr_schedule)
}

Write-Host "Saved summary CSV: $summaryPath" -ForegroundColor Magenta

