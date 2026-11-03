[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Prepare', 'Tune', 'Match', 'Calibrate', 'All', 'Status')]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [string]$Pgn,
    [string]$CurrentParams,
    [string]$Openings,
    [string]$EnginePath,
    [string]$TunerPath,
    [string]$CutechessPath,
    [string]$PythonPath = 'python',
    [string]$RunRoot,

    [ValidateRange(1, 99)]
    [int]$ValidationPercent = 10,
    [ValidateRange(0, 2147483647)]
    [int]$Seed = 1,
    [ValidateRange(1, 2147483647)]
    [int]$Iterations = 300,
    [ValidateRange(1, 2147483647)]
    [int]$Batch = 4096,
    [ValidateSet('all', 'material', 'piece-values')]
    [string]$Family = 'all',
    [ValidateRange(0.000001, 1000)]
    [double]$K = 1.13,
    [ValidateRange(2, 1000000)]
    [int]$Games = 10000,
    [ValidateRange(1, 256)]
    [int]$Concurrency = 1,
    [ValidateRange(1, 100)]
    [int]$PairsPerBatch = 10,
    [ValidateRange(1, 65536)]
    [int]$HashMb = 64,
    [ValidateRange(0, 5000)]
    [int]$MoveOverhead = 10,
    [string]$TimeControl = '10+0.1',
    [ValidateRange(1, 60)]
    [int]$OpeningPlies = 16,
    [switch]$Promote
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunRoot) {
    $RunRoot = Join-Path $repoRoot 'experiments\TriangleTest'
}
if (-not $EnginePath) {
    $EnginePath = Join-Path $repoRoot 'build\windows\x64\release\bermuda.exe'
}
if (-not $TunerPath) {
    $TunerPath = Join-Path $repoRoot 'build\windows\x64\release\bermuda_tuner.exe'
}
if (-not $CutechessPath) {
    $CutechessPath = Join-Path $repoRoot 'cutechess-1.4.0-win64\cutechess-cli.exe'
}

if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$') {
    throw "RunId must be 1-80 safe filename characters."
}
if (($Games % 2) -ne 0) {
    throw 'Games must be even so each paired opening is played with colors reversed.'
}

$runDir = Join-Path $RunRoot $RunId
$manifestPath = Join-Path $runDir 'triangle-manifest.json'
$trainPath = Join-Path $runDir 'train.data'
$validationPath = Join-Path $runDir 'validation.data'
$candidatePath = Join-Path $runDir 'candidate-params.txt'
$checkpointPath = Join-Path $runDir 'optimizer.checkpoint'
$matchDir = Join-Path $runDir 'matches'
$aggregatePgn = Join-Path $runDir 'paired-match.pgn'
$reproducibilityPath = Join-Path $runDir 'reproducibility.json'
$sprtAnalyzerPath = Join-Path $PSScriptRoot 'triangle_sprt.py'
$sprtAlpha = 0.05
$sprtBeta = 0.05
$sprtElo0 = 0.0
$sprtElo1 = 5.0

function Get-FileFingerprint([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Resolve-RequiredFile([string]$Path, [string]$Name) {
    if (-not $Path) {
        throw "$Name is required for this stage."
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Name not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Save-Manifest {
    $tempPath = "$manifestPath.tmp"
    $json = $manifest | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($tempPath, $json, [System.Text.Encoding]::UTF8)
    Move-Item -LiteralPath $tempPath -Destination $manifestPath -Force
}

function Get-Manifest {
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Run '$RunId' has no manifest yet. Start with -Stage Prepare or -Stage All."
    }
    return (Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json)
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Executable not found: $Executable"
    }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable"
    }
}

function Get-SprtResult([int]$CompletedPairs, [int]$MaximumPairs, $PreviousResult = $null) {
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $arguments = @(
        $sprtAnalyzerPath,
        '--matches-dir', $matchDir,
        '--completed-pairs', "$CompletedPairs",
        '--max-pairs', "$MaximumPairs",
        '--alpha', $sprtAlpha.ToString('R', $culture),
        '--beta', $sprtBeta.ToString('R', $culture),
        '--elo0', $sprtElo0.ToString('R', $culture),
        '--elo1', $sprtElo1.ToString('R', $culture)
    )
    if ($null -ne $PreviousResult) {
        $previousJson = $PreviousResult | ConvertTo-Json -Compress -Depth 5
        $previousEncoded = [Convert]::ToBase64String(
            [System.Text.Encoding]::UTF8.GetBytes($previousJson)
        )
        $arguments += @('--previous-result-base64', $previousEncoded)
    }
    $output = & $PythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "TriangleTest SPRT analysis failed with exit code $LASTEXITCODE."
    }
    try {
        return ($output | ConvertFrom-Json)
    } catch {
        throw "TriangleTest SPRT analyzer returned invalid JSON: $($_.Exception.Message)"
    }
}

function Write-MatchReport($SprtResult) {
    $promotionEligible = $SprtResult.status -eq 'PASS'
    $score = 0.0
    if ($SprtResult.completed_games -gt 0) {
        $score = $SprtResult.candidate_score / $SprtResult.completed_games
    }
    $llrText = 'not available'
    if ($null -ne $SprtResult.llr) {
        $llrText = [Math]::Round([double]$SprtResult.llr, 4)
    }
    $report = @"
TriangleTest paired-match SPRT report
Decision: $($SprtResult.status)
Candidate score: $($SprtResult.candidate_wins)-$($SprtResult.candidate_losses)-$($SprtResult.candidate_draws)
Completed games/pairs: $($SprtResult.completed_games)/$($SprtResult.completed_pairs)
Maximum games: $($SprtResult.max_games)
Score fraction: $([Math]::Round($score, 4))
SPRT: H0=$sprtElo0 Elo, H1=$sprtElo1 Elo, alpha=$sprtAlpha, beta=$sprtBeta (logistic model)
LLR: $llrText (base boundaries $([Math]::Round([double]$SprtResult.lower_boundary, 4)) to $([Math]::Round([double]$SprtResult.upper_boundary, 4)); adjusted boundaries $([Math]::Round([double]$SprtResult.adjusted_lower_boundary, 4)) to $([Math]::Round([double]$SprtResult.adjusted_upper_boundary, 4)))
Pentanomial pair counts (LL, LD+DL, LW+DD+WL, DW+WD, WW): $($SprtResult.pentanomial_counts -join ', ')
Promotion eligible (SPRT PASS): $promotionEligible

PASS means the upper SPRT boundary was crossed in favor of H1; FAIL means the
lower boundary was crossed against H1. INCONCLUSIVE at the game limit is not a
pass. The paired-match PGN is retained for independent Ordo analysis.
"@
    Set-Content -LiteralPath (Join-Path $runDir 'promotion-report.txt') -Value $report -Encoding UTF8
    Write-Host $report
    return $promotionEligible
}

if ($Stage -eq 'Status') {
    $manifest = Get-Manifest
    $manifest | Format-List
    return
}

if ($Stage -eq 'Prepare' -or $Stage -eq 'All') {
    $Pgn = Resolve-RequiredFile $Pgn 'Pgn'
    $currentHash = Get-FileFingerprint $Pgn
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $manifest = [ordered]@{
            schema = 2
            runId = $RunId
            created = (Get-Date).ToString('o')
            sourcePgn = $Pgn
            sourcePgnSha256 = $currentHash
            splitSeed = $Seed
            validationPercent = $ValidationPercent
            prepared = $false
            tuned = $false
            candidateSha256 = ''
            tuningOptions = $null
            matchOptions = $null
            completedPairs = 0
            gamesRequested = 0
            sprtStatus = 'NOT_STARTED'
            sprtLLR = $null
            sprtLowerBoundary = $null
            sprtUpperBoundary = $null
            sprtAdjustedLowerBoundary = $null
            sprtAdjustedUpperBoundary = $null
            sprtOvershootState = $null
            pentanomialCounts = @(0, 0, 0, 0, 0)
            pairSha256 = @()
            reproducibility = $null
            promoted = $false
            promotedParamsSha256 = ''
        }
        Save-Manifest
    } else {
        $manifest = Get-Manifest
        if ($manifest.sourcePgnSha256 -ne $currentHash -or $manifest.splitSeed -ne $Seed -or $manifest.validationPercent -ne $ValidationPercent) {
            throw 'Preparation inputs differ from this run manifest. Use a new RunId for changed inputs.'
        }
    }
    if (-not $manifest.prepared) {
        $prepareArgs = @(
            '--prepare-pgn', $Pgn,
            '--train-out', $trainPath,
            '--validation-out', $validationPath,
            '--validation-percent', "$ValidationPercent",
            '--split-seed', "$Seed"
        )
        Invoke-Checked $TunerPath $prepareArgs
        $manifest.prepared = $true
        $manifest.trainSha256 = Get-FileFingerprint $trainPath
        $manifest.validationSha256 = Get-FileFingerprint $validationPath
        Save-Manifest
    }
    if ($Stage -eq 'Prepare') {
        Write-Output "Prepared datasets saved under $runDir"
        return
    }
}

if ($Stage -eq 'Tune' -or $Stage -eq 'All') {
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw 'Prepare the run before tuning.'
    }
    $manifest = Get-Manifest
    if (-not $manifest.prepared) {
        throw 'Datasets are not prepared.'
    }
    if ($manifest.trainSha256 -ne (Get-FileFingerprint $trainPath) -or $manifest.validationSha256 -ne (Get-FileFingerprint $validationPath)) {
        throw 'Prepared dataset files changed since preparation.'
    }
    $CurrentParams = Resolve-RequiredFile $CurrentParams 'CurrentParams'
    if ($manifest.tuningOptions) {
        $prior = $manifest.tuningOptions
        if ($prior.currentParamsSha256 -ne (Get-FileFingerprint $CurrentParams) -or $prior.iterations -ne $Iterations -or $prior.batch -ne $Batch -or $prior.family -ne $Family -or $prior.k -ne $K -or $prior.seed -ne $Seed) {
            throw 'Tuning options differ from this run checkpoint. Use the same options or create a new RunId.'
        }
    } else {
        $manifest.tuningOptions = [ordered]@{
            currentParamsSha256 = Get-FileFingerprint $CurrentParams
            iterations = $Iterations
            batch = $Batch
            family = $Family
            k = $K
            seed = $Seed
        }
        Save-Manifest
    }
    if (-not $manifest.tuned -or -not (Test-Path -LiteralPath $candidatePath)) {
        $tuneArgs = @(
            '--train-data', $trainPath,
            '--validation-data', $validationPath,
            '--init', $CurrentParams,
            '--out', $candidatePath,
            '--checkpoint', $checkpointPath,
            '--iterations', "$Iterations",
            '--batch', "$Batch",
            '--seed', "$Seed",
            '--family', $Family,
            '--k', "$K"
        )
        if (Test-Path -LiteralPath $checkpointPath) {
            $tuneArgs += '--resume'
        }
        Invoke-Checked $TunerPath $tuneArgs
        $manifest.tuned = $true
        $manifest.candidateSha256 = Get-FileFingerprint $candidatePath
        Save-Manifest
    } elseif ($manifest.candidateSha256 -ne (Get-FileFingerprint $candidatePath)) {
        throw 'Candidate file changed since the manifest was written.'
    }
    if ($Stage -eq 'Tune') {
        Write-Output "Candidate parameters saved to $candidatePath"
        return
    }
}

if ($Stage -eq 'Calibrate') {
    if ($Promote) {
        throw '-Promote is not valid for an identical-baseline calibration run.'
    }
    $CurrentParams = Resolve-RequiredFile $CurrentParams 'CurrentParams'
    $Openings = Resolve-RequiredFile $Openings 'Openings'
    $EnginePath = Resolve-RequiredFile $EnginePath 'EnginePath'
    $CutechessPath = Resolve-RequiredFile $CutechessPath 'CutechessPath'
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Manifest
        if ($manifest.runType -ne 'identical-baseline-calibration') {
            throw "RunId '$RunId' belongs to a non-calibration run. Use a new RunId."
        }
    } else {
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $currentParamsHash = Get-FileFingerprint $CurrentParams
        $manifest = [ordered]@{
            schema = 2
            runId = $RunId
            runType = 'identical-baseline-calibration'
            created = (Get-Date).ToString('o')
            prepared = $true
            tuned = $true
            candidateSha256 = $currentParamsHash
            tuningOptions = @{ currentParamsSha256 = $currentParamsHash }
            matchOptions = $null
            completedPairs = 0
            gamesRequested = 0
            sprtStatus = 'NOT_STARTED'
            sprtLLR = $null
            sprtLowerBoundary = $null
            sprtUpperBoundary = $null
            sprtAdjustedLowerBoundary = $null
            sprtAdjustedUpperBoundary = $null
            sprtOvershootState = $null
            pentanomialCounts = @(0, 0, 0, 0, 0)
            pairSha256 = @()
            reproducibility = $null
            promoted = $false
            promotedParamsSha256 = ''
        }
        Save-Manifest
    }
}

if ($Stage -eq 'Match' -or $Stage -eq 'Calibrate' -or $Stage -eq 'All') {
    $manifest = Get-Manifest
    if ($manifest.schema -ne 2) {
        throw "Run manifest schema '$($manifest.schema)' predates SPRT support. Start a new RunId for strength testing."
    }
    foreach ($requiredProperty in @(
        'sprtStatus',
        'sprtLLR',
        'sprtLowerBoundary',
        'sprtUpperBoundary',
        'sprtAdjustedLowerBoundary',
        'sprtAdjustedUpperBoundary',
        'sprtOvershootState',
        'pentanomialCounts',
        'pairSha256'
    )) {
        if ($manifest.PSObject.Properties.Name -notcontains $requiredProperty) {
            throw "Run manifest is missing SPRT state '$requiredProperty'. Start a new RunId."
        }
    }
    $candidateForMatch = $candidatePath
    if ($Stage -eq 'Calibrate') {
        $candidateForMatch = $CurrentParams
    }
    if (-not $manifest.tuned -or (-not (Test-Path -LiteralPath $candidateForMatch) -and $Stage -ne 'Calibrate')) {
        throw 'Tune the run before strength evaluation.'
    }
    $CurrentParams = Resolve-RequiredFile $CurrentParams 'CurrentParams'
    $Openings = Resolve-RequiredFile $Openings 'Openings'
    $EnginePath = Resolve-RequiredFile $EnginePath 'EnginePath'
    $CutechessPath = Resolve-RequiredFile $CutechessPath 'CutechessPath'
    if (-not (Test-Path -LiteralPath $sprtAnalyzerPath -PathType Leaf)) {
        throw "SPRT analyzer not found: $sprtAnalyzerPath"
    }
    $pythonCommand = Get-Command -Name $PythonPath -CommandType Application -ErrorAction Stop |
        Select-Object -First 1
    $PythonPath = $pythonCommand.Source
    $pythonVersion = (& $PythonPath --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $pythonVersion.StartsWith('Python 3.')) {
        throw "TriangleTest requires Python 3; '$PythonPath --version' returned '$pythonVersion'."
    }
    $analyzerHash = Get-FileFingerprint $sprtAnalyzerPath
    $environmentInfo = [ordered]@{
        operatingSystem = [Environment]::OSVersion.VersionString
        processorIdentifier = $env:PROCESSOR_IDENTIFIER
        processorArchitecture = $env:PROCESSOR_ARCHITECTURE
        logicalProcessorCount = [Environment]::ProcessorCount
        powershellVersion = $PSVersionTable.PSVersion.ToString()
        dotNetVersion = [Environment]::Version.ToString()
    }
    if ($manifest.candidateSha256 -ne (Get-FileFingerprint $candidateForMatch)) {
        throw 'Candidate file changed since tuning was recorded.'
    }
    if ($manifest.tuningOptions.currentParamsSha256 -ne (Get-FileFingerprint $CurrentParams)) {
        throw 'Match baseline must be the same parameter file used to initialize tuning.'
    }
    if ($manifest.gamesRequested -eq 0) {
        $manifest.gamesRequested = $Games
        Save-Manifest
    } elseif ($manifest.gamesRequested -ne $Games) {
        throw 'Requested match size differs from this run. Use the original game count or a new RunId.'
    }
    if ($manifest.matchOptions) {
        $prior = $manifest.matchOptions
        if ($prior.baseParamsSha256 -ne (Get-FileFingerprint $CurrentParams) -or $prior.openingsSha256 -ne (Get-FileFingerprint $Openings) -or $prior.engineSha256 -ne (Get-FileFingerprint $EnginePath) -or $prior.cutechessSha256 -ne (Get-FileFingerprint $CutechessPath) -or $prior.timeControl -ne $TimeControl -or $prior.concurrency -ne $Concurrency -or $prior.openingPlies -ne $OpeningPlies -or $prior.openingSeed -ne $Seed -or $prior.pairsPerBatch -ne $PairsPerBatch -or $prior.hashMb -ne $HashMb -or $prior.moveOverhead -ne $MoveOverhead -or $prior.sprtAnalyzerSha256 -ne $analyzerHash -or $prior.pythonVersion -ne $pythonVersion -or $prior.sprtAlpha -ne $sprtAlpha -or $prior.sprtBeta -ne $sprtBeta -or $prior.sprtElo0 -ne $sprtElo0 -or $prior.sprtElo1 -ne $sprtElo1 -or $prior.operatingSystem -ne $environmentInfo.operatingSystem -or $prior.processorIdentifier -ne $environmentInfo.processorIdentifier -or $prior.processorArchitecture -ne $environmentInfo.processorArchitecture -or $prior.logicalProcessorCount -ne $environmentInfo.logicalProcessorCount -or $prior.powershellVersion -ne $environmentInfo.powershellVersion -or $prior.dotNetVersion -ne $environmentInfo.dotNetVersion) {
            throw 'Match configuration differs from this run. Use a new RunId for changed settings.'
        }
    } else {
        $manifest.matchOptions = [ordered]@{
            baseParamsSha256 = Get-FileFingerprint $CurrentParams
            openingsSha256 = Get-FileFingerprint $Openings
            engineSha256 = Get-FileFingerprint $EnginePath
            cutechessSha256 = Get-FileFingerprint $CutechessPath
            sprtAnalyzerSha256 = $analyzerHash
            pythonVersion = $pythonVersion
            sprtAlpha = $sprtAlpha
            sprtBeta = $sprtBeta
            sprtElo0 = $sprtElo0
            sprtElo1 = $sprtElo1
            openingSeed = $Seed
            pairsPerBatch = $PairsPerBatch
            hashMb = $HashMb
            moveOverhead = $MoveOverhead
            operatingSystem = $environmentInfo.operatingSystem
            processorIdentifier = $environmentInfo.processorIdentifier
            processorArchitecture = $environmentInfo.processorArchitecture
            logicalProcessorCount = $environmentInfo.logicalProcessorCount
            powershellVersion = $environmentInfo.powershellVersion
            dotNetVersion = $environmentInfo.dotNetVersion
            timeControl = $TimeControl
            concurrency = $Concurrency
            openingPlies = $OpeningPlies
        }
        Save-Manifest
    }

    $manifest.reproducibility = [ordered]@{
        runId = $RunId
        runType = $(if ($Stage -eq 'Calibrate') { 'identical-baseline-calibration' } else { 'candidate-match' })
        recordedAt = (Get-Date).ToString('o')
        operatingSystem = $environmentInfo.operatingSystem
        processorIdentifier = $environmentInfo.processorIdentifier
        processorArchitecture = $environmentInfo.processorArchitecture
        logicalProcessorCount = $environmentInfo.logicalProcessorCount
        powershellVersion = $environmentInfo.powershellVersion
        dotNetVersion = $environmentInfo.dotNetVersion
        pythonExecutable = $PythonPath
        pythonVersion = $pythonVersion
        enginePath = $EnginePath
        engineSha256 = $manifest.matchOptions.engineSha256
        cutechessPath = $CutechessPath
        cutechessSha256 = $manifest.matchOptions.cutechessSha256
        openingsPath = $Openings
        openingsSha256 = $manifest.matchOptions.openingsSha256
        baselineParamsPath = $CurrentParams
        baselineParamsSha256 = $manifest.matchOptions.baseParamsSha256
        candidateParamsPath = $candidateForMatch
        candidateParamsSha256 = $manifest.candidateSha256
        timeControl = $TimeControl
        hashMb = $HashMb
        moveOverhead = $MoveOverhead
        concurrency = $Concurrency
        pairsPerBatch = $PairsPerBatch
        openingPlies = $OpeningPlies
        openingSeed = $Seed
        sprtAlpha = $sprtAlpha
        sprtBeta = $sprtBeta
        sprtElo0 = $sprtElo0
        sprtElo1 = $sprtElo1
        sprtModel = 'logistic'
        sprtAnalyzerSha256 = $analyzerHash
    }
    $metadataTempPath = "$reproducibilityPath.tmp"
    [System.IO.File]::WriteAllText(
        $metadataTempPath,
        ($manifest.reproducibility | ConvertTo-Json -Depth 6),
        [System.Text.Encoding]::UTF8
    )
    Move-Item -LiteralPath $metadataTempPath -Destination $reproducibilityPath -Force
    Save-Manifest

    New-Item -ItemType Directory -Force -Path $matchDir | Out-Null
    $pairsRequested = [int]($Games / 2)
    if (@($manifest.pairSha256).Count -ne [int]$manifest.completedPairs) {
        throw 'Run manifest paired-PGN hashes do not match the completed-pair count.'
    }
    for ($i = 1; $i -le [int]$manifest.completedPairs; $i++) {
        $pairFile = Join-Path $matchDir ('pair-{0:D5}.pgn' -f $i)
        if ($manifest.pairSha256[$i - 1] -ne (Get-FileFingerprint $pairFile)) {
            throw "Completed pair $i changed since it was recorded."
        }
    }
    $sprtResult = Get-SprtResult ([int]$manifest.completedPairs) $pairsRequested
    $manifest.sprtStatus = $sprtResult.status
    $manifest.sprtLLR = $sprtResult.llr
    $manifest.sprtLowerBoundary = $sprtResult.lower_boundary
    $manifest.sprtUpperBoundary = $sprtResult.upper_boundary
    $manifest.sprtAdjustedLowerBoundary = $sprtResult.adjusted_lower_boundary
    $manifest.sprtAdjustedUpperBoundary = $sprtResult.adjusted_upper_boundary
    $manifest.sprtOvershootState = $sprtResult.overshoot_state
    $manifest.pentanomialCounts = $sprtResult.pentanomial_counts
    Save-Manifest
    while ($manifest.completedPairs -lt $pairsRequested -and $sprtResult.status -eq 'CONTINUE') {
        $pairNumber = [int]$manifest.completedPairs + 1
        $batchPairCount = [Math]::Min($PairsPerBatch, $pairsRequested - $pairNumber + 1)
        $batchLastPair = $pairNumber + $batchPairCount - 1
        $openingSeed = [int]((([long]$Seed + $pairNumber - 1) % 2147483647) + 1)
        $batchName = 'batch-{0:D5}-{1:D5}.pgn' -f $pairNumber, $batchLastPair
        $batchPath = Join-Path $matchDir $batchName
        if (Test-Path -LiteralPath $batchPath) {
            $archive = Join-Path $matchDir ('retry-{0:D5}-{1:D5}-{2}.pgn' -f
                $pairNumber, $batchLastPair, (Get-Date -Format 'yyyyMMddHHmmss'))
            Move-Item -LiteralPath $batchPath -Destination $archive
        }
        $matchArgs = @(
            '-srand', "$openingSeed",
            '-engine', 'name=Baseline', "cmd=$EnginePath", 'proto=uci', "option.EvalParams=$CurrentParams", "option.Hash=$HashMb", "option.MoveOverhead=$MoveOverhead",
            '-engine', 'name=Candidate', "cmd=$EnginePath", 'proto=uci', "option.EvalParams=$candidateForMatch", "option.Hash=$HashMb", "option.MoveOverhead=$MoveOverhead",
            '-each', "tc=$TimeControl",
            '-openings', "file=$Openings", 'format=pgn', 'order=random',
            "plies=$OpeningPlies", 'policy=round',
            '-games', '2',
            '-rounds', "$batchPairCount",
            '-concurrency', "$Concurrency",
            '-recover',
            '-pgnout', $batchPath
        )
        $logPath = Join-Path $matchDir ('batch-{0:D5}-{1:D5}.log' -f $pairNumber, $batchLastPair)
        $previousErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $CutechessPath @matchArgs 2>&1 | Tee-Object -FilePath $logPath
        $ErrorActionPreference = $previousErrorPreference
        if ($LASTEXITCODE -ne 0) {
            throw "Cutechess failed in paired batch $pairNumber-$batchLastPair with exit code $LASTEXITCODE. Resume with the same RunId."
        }
        if (-not (Test-Path -LiteralPath $batchPath -PathType Leaf)) {
            throw "Cutechess did not write the expected PGN batch $pairNumber-$batchLastPair."
        }
        $splitArgs = @(
            $sprtAnalyzerPath,
            '--split-pgn', $batchPath,
            '--output-dir', $matchDir,
            '--first-pair', "$pairNumber",
            '--expected-pairs', "$batchPairCount"
        )
        & $PythonPath @splitArgs
        if ($LASTEXITCODE -ne 0) {
            throw "SPRT analyzer could not validate paired batch $pairNumber-$batchLastPair."
        }
        for ($currentPair = $pairNumber; $currentPair -le $batchLastPair; $currentPair++) {
            $pairPath = Join-Path $matchDir ('pair-{0:D5}.pgn' -f $currentPair)
            $sprtResult = Get-SprtResult $currentPair $pairsRequested $sprtResult
            $manifest.completedPairs = $currentPair
            $manifest.sprtStatus = $sprtResult.status
            $manifest.sprtLLR = $sprtResult.llr
            $manifest.sprtLowerBoundary = $sprtResult.lower_boundary
            $manifest.sprtUpperBoundary = $sprtResult.upper_boundary
            $manifest.sprtAdjustedLowerBoundary = $sprtResult.adjusted_lower_boundary
            $manifest.sprtAdjustedUpperBoundary = $sprtResult.adjusted_upper_boundary
            $manifest.sprtOvershootState = $sprtResult.overshoot_state
            $manifest.pentanomialCounts = $sprtResult.pentanomial_counts
            $manifest.pairSha256 = @($manifest.pairSha256) + @(Get-FileFingerprint $pairPath)
            Save-Manifest
            Write-Output "Recorded opening pair $currentPair of $pairsRequested; SPRT $($sprtResult.status), LLR $([Math]::Round([double]$sprtResult.llr, 4))."
            if ($sprtResult.status -ne 'CONTINUE') {
                for ($unusedPair = $currentPair + 1; $unusedPair -le $batchLastPair; $unusedPair++) {
                    $unusedPath = Join-Path $matchDir ('pair-{0:D5}.pgn' -f $unusedPair)
                    if (Test-Path -LiteralPath $unusedPath) {
                        $unusedArchive = Join-Path $matchDir ('unused-after-stop-{0:D5}.pgn' -f $unusedPair)
                        Move-Item -LiteralPath $unusedPath -Destination $unusedArchive -Force
                    }
                }
                break
            }
        }
    }

    $builder = New-Object System.Text.StringBuilder
    for ($i = 1; $i -le [int]$manifest.completedPairs; $i++) {
        $pairFile = Join-Path $matchDir ('pair-{0:D5}.pgn' -f $i)
        [void]$builder.AppendLine((Get-Content -LiteralPath $pairFile -Raw).TrimEnd())
        [void]$builder.AppendLine()
    }
    [System.IO.File]::WriteAllText($aggregatePgn, $builder.ToString(),
                                   [System.Text.Encoding]::UTF8)
    $promotionEligible = Write-MatchReport $sprtResult
    if ($Promote -and $Stage -eq 'Calibrate') {
        throw 'Calibration runs cannot promote parameters.'
    }
    if ($Promote) {
        if (-not $promotionEligible) {
            throw "Promotion gate was not met. Candidate remains unpromoted; see promotion-report.txt."
        }
        $promotedPath = Join-Path $runDir 'promoted-params.txt'
        Copy-Item -LiteralPath $candidatePath -Destination $promotedPath
        $manifest.promoted = $true
        $manifest.promotedParamsSha256 = Get-FileFingerprint $promotedPath
        Save-Manifest
        Write-Output "Candidate passed the SPRT and was copied to $promotedPath. Baseline was not overwritten."
    }
    Write-Output "Combined PGN for Ordo or further analysis: $aggregatePgn"
}
