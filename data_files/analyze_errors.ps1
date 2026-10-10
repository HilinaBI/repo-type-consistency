$ErrorActionPreference = "Stop"

# Paths
$inputJson  = "C:\Users\hp\Downloads\research\thesis pipeline\data_files\mypy_detailed_report.json"
$outputTxt  = "C:\Users\hp\Downloads\research\thesis pipeline\data_files\error_analysis_report.txt"

# Load the JSON
$json = Get-Content $inputJson -Raw | ConvertFrom-Json

# Resolvability table
$resolvability = @{
    "name-defined"          = "code-fixable"
    "attr-defined"          = "code-fixable"
    "arg-type"              = "code-fixable"
    "assignment"            = "code-fixable"
    "return-value"          = "code-fixable"
    "call-arg"              = "code-fixable"
    "call-overload"         = "code-fixable"
    "union-attr"            = "code-fixable"
    "operator"              = "code-fixable"
    "index"                 = "code-fixable"
    "no-redef"              = "code-fixable"
    "override"              = "code-fixable"
    "var-annotated"         = "code-fixable"
    "type-var"              = "code-fixable"
    "has-type"              = "code-fixable"
    "used-before-def"       = "code-fixable"
    "misc"                  = "code-fixable"
    "list-item"             = "code-fixable"
    "dict-item"             = "code-fixable"
    "typeddict-item"        = "code-fixable"
    "literal-required"      = "code-fixable"
    "type-abstract"         = "code-fixable"
    "str-bytes-safe"        = "code-fixable"
    "str-format"            = "code-fixable"
    "str-unpack"            = "code-fixable"
    "truthy-function"       = "code-fixable"
    "method-assign"         = "code-fixable"
    "empty-body"            = "code-fixable"
    "return"                = "code-fixable"
    "valid-type"            = "code-fixable"
    "func-returns-value"    = "code-fixable"
    "type-arg"              = "code-fixable"
    "import-untyped"        = "environment"
    "import-not-found"      = "environment"
    "overload-cannot-match" = "structural"
    "syntax"                = "structural"
}

# Collect all errors across all repos
$allErrors = @()
foreach ($split in @("training","validation","test")) {
    $repos = $json.by_split.$split
    foreach ($repoName in $repos.PSObject.Properties.Name) {
        $repo = $repos.$repoName
        if (-not $repo.final_errors) { continue }
        foreach ($e in $repo.final_errors) {
            $allErrors += [pscustomobject]@{
                split  = $split
                repo   = $repoName
                file   = $e.file
                line   = $e.line
                code   = $e.code
                msg    = $e.message
                ac     = $e.annotation_caused
            }
        }
    }
}

# Global totals
$totalErrors = $allErrors.Count
$totalAnnotationCaused = ($allErrors | Where-Object { $_.ac }).Count
$totalNotAnnotationCaused = $totalErrors - $totalAnnotationCaused

# By code
$byCode = $allErrors | Group-Object code | Sort-Object Count -Descending

# By resolvability
$byResolvability = @{}
foreach ($e in $allErrors) {
    $tag = $resolvability[$e.code]
    if (-not $tag) { $tag = "other" }
    if (-not $byResolvability.ContainsKey($tag)) { $byResolvability[$tag] = 0 }
    $byResolvability[$tag]++
}

# Write the report
$lines = @()
$lines += "ERROR ANALYSIS REPORT"
$lines += "Generated from: mypy_detailed_report.json"
$lines += ""

$lines += "GLOBAL TOTALS"
$lines += "  Total remaining errors          : $totalErrors"
$lines += "  Annotation-caused (of the total): $totalAnnotationCaused"
$lines += "  Not annotation-caused           : $totalNotAnnotationCaused"
$lines += ""

$lines += "RESOLVABILITY SUMMARY"
foreach ($k in @("code-fixable","environment","structural","other")) {
    if ($byResolvability.ContainsKey($k)) {
        $lines += "  $k : $($byResolvability[$k])"
    } else {
        $lines += "  $k : 0"
    }
}
$lines += ""

$lines += "ERRORS BY CODE"
$lines += ""
foreach ($group in $byCode) {
    $code = $group.Name
    $count = $group.Count
    $tag = $resolvability[$code]
    if (-not $tag) { $tag = "other" }
    $desc = $json.code_descriptions.$code
    if (-not $desc) { $desc = "No description" }

    $lines += "[$code]  count=$count  resolvability=$tag"
    $lines += "    description: $desc"

    # Repos
    $repos = ($group.Group | Select-Object -ExpandProperty repo -Unique) -join ", "
    $lines += "    repos: $repos"

    # Examples (up to 2)
    $examples = $group.Group | Select-Object -First 2
    $i = 1
    foreach ($ex in $examples) {
        $lines += "    example $i : $($ex.file):$($ex.line)"
        $lines += "                 $($ex.msg)"
        $i++
    }
    $lines += ""
}

$lines += "PER-REPO SUMMARY"
foreach ($split in @("training","validation","test")) {
    $lines += "  Split: $split"
    $repos = $json.by_split.$split
    foreach ($repoName in $repos.PSObject.Properties.Name) {
        $repo = $repos.$repoName
        if (-not $repo.final_total) {
            $lines += "    $repoName : 0"
            continue
        }
        $lines += "    $repoName : final=$($repo.final_total)  removed=$($repo.removed_annotations)"
    }
    $lines += ""
}

$lines | Set-Content $outputTxt -Encoding UTF8
Write-Host "Report written to: $outputTxt"