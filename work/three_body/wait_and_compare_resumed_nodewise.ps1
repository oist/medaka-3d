param(
    [string]$Repository = "C:\Users\machi\Documents\GitHub\medaka-3d"
)

$ErrorActionPreference = "Stop"
$trackerRoot = Join-Path $Repository "work\three_body\nodewise_body_row_five_minute_resume_099600_v1"
$trackerSeal = Join-Path $trackerRoot "tracking.provenance.json"
$comparisonRoot = Join-Path $Repository "work\three_body\resumed_nodewise_five_minute_comparison_v1"
$statusPath = Join-Path $Repository "work\three_body\wait_and_compare_resumed_nodewise.status.json"
$python = Join-Path $Repository ".venv\Scripts\python.exe"
$comparator = Join-Path $Repository "work\three_body\compare_resumed_nodewise_five_minutes.py"

while (-not (Test-Path -LiteralPath $trackerSeal)) {
    Start-Sleep -Seconds 60
}

$seal = Get-Content -LiteralPath $trackerSeal -Raw | ConvertFrom-Json
if ($seal.status -ne "ok") {
    @{
        status = "tracking_failed"
        tracker_seal = $trackerSeal
        observed_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
    exit 2
}

if (Test-Path -LiteralPath $comparisonRoot) {
    throw "Comparison output must be fresh: $comparisonRoot"
}

$digest = (Get-FileHash -LiteralPath $trackerSeal -Algorithm SHA256).Hash.ToLowerInvariant()
& $python -B $comparator --nodewise-root $trackerRoot --nodewise-sha $digest --output $comparisonRoot
$exitCode = $LASTEXITCODE
@{
    status = if ($exitCode -eq 0) { "comparison_finished" } else { "comparison_failed" }
    exit_code = $exitCode
    tracker_seal_sha256 = $digest
    observed_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
exit $exitCode
