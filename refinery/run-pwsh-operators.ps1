<#
.SYNOPSIS
Collect what PowerShell's operators and conversions actually produce, as a grid.

.DESCRIPTION
Writes pwsh-operators.json into the output directory. This is a separate script from run-pwsh.ps1
on purpose, and it writes exactly one file: the five tables that script produces are pinned, and
regenerating them here would rewrite the command and alias tables with whatever modules happen to
be installed on the collecting machine. A script that cannot open those files cannot do that.

The subject is also different. run-pwsh.ps1 collects what a *host* has — its commands, its aliases,
its loaded types. This collects what the *language* does, which is fixed for a given version of
PowerShell and does not depend on what is installed.

A cell is not a type. `(op, left type, right type)` does not determine a result type: 512MB * 512MB
is a Double from the same Int32-by-Int32 cell that gives an Int32 elsewhere, and `12 + '0xabc'` is
2760 where `16 + 'file'` throws. So each cell records the *set* of result types observed over
several witness values per type, and whether any pair threw. A cell whose set has one entry and no
throw is type-determined; anything else is decided by the values and needs a kernel, which is
written by hand and is not this script's business to guess.

Witnesses are chosen for the case splits the specification makes — zero, one, a negative, the
extremes of the range, a string that parses as a number and one that does not — rather than for
coverage of the grid, because it is those splits that make a cell value-dependent.

Nothing here reads a file, starts a process, or touches the network: every operand is a literal
written in this script.

.PARAMETER Unauthoritative
Collect on a host that is not Windows PowerShell 5.1, stamping the result accordingly and writing
to a temporary directory instead of the data directory.

.PARAMETER Culture
The culture to collect under. The grid is expected to be culture-free; collecting twice and
comparing is what turns that expectation into a measurement, and a cell that moves has to be
refused or parameterised rather than shipped.
#>
[CmdletBinding()]
param(
    [switch] $Unauthoritative,
    [string] $Culture,
    [string] $OutputDirectory
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

$ScriptDir =
    if ($PSScriptRoot) { $PSScriptRoot }
    elseif ($PSCommandPath) { Split-Path -Parent $PSCommandPath }
    else { (Get-Location).Path }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $ScriptDir 'data' }

$SchemaVersion = 1
$JsonDepth = 64

if ($Culture) {
    [System.Threading.Thread]::CurrentThread.CurrentCulture =
        [System.Globalization.CultureInfo]::GetCultureInfo($Culture)
}

#: One entry per type the grid is over, holding expressions that evaluate to that type. The seed is
#: the literal types, every integer type a cast may target, and the two shapes that are not scalars
#: at all — a collection and $null — because an operator's behaviour over those is what the ledgered
#: defects turn on.
#:
#: A collection's own splits are its length and whether its elements are collections: a length is
#: how much an operation has to reach into — none, one, or several — and an element that is itself a
#: collection is one an operation may have no answer for. Which operators actually branch on either
#: is the measurement rather than the reason for taking it. A scalar cast turns out to throw at
#: every length including one, so `[int] @(5)` is no better off than `[int] @(1, 2)`; and the
#: ordering comparisons throw only where an element is a collection, which is a throw the row had no
#: witness for while every collection in it was flat.
$Witnesses = [ordered] @{
    'System.Byte'    = @('[byte]0', '[byte]1', '[byte]255')
    'System.SByte'   = @('[sbyte]0', '[sbyte]1', '[sbyte]-128', '[sbyte]127')
    'System.Int16'   = @('[int16]0', '[int16]1', '[int16]-32768', '[int16]32767')
    'System.UInt16'  = @('[uint16]0', '[uint16]1', '[uint16]65535')
    'System.Int32'   = @('0', '1', '-1', '2147483647', '-2147483648')
    'System.UInt32'  = @('[uint32]0', '[uint32]1', '[uint32]4294967295')
    'System.Int64'   = @('0L', '1L', '-1L', '9223372036854775807L')
    'System.UInt64'  = @('[uint64]0', '[uint64]1', '[uint64]18446744073709551615')
    'System.Single'  = @('[single]0', '[single]1.5', '[single]-1.5')
    'System.Double'  = @('0.0', '1.5', '-1.5', '[double]::MaxValue')
    'System.Decimal' = @('0d', '1.5d', '-1.5d', '[decimal]::MaxValue')
    'System.String'  = @("''", "'abc'", "'5'", "'0xabc'", "'-2'")
    'System.Char'    = @('[char]65', '[char]0', '[char]48')
    'System.Boolean' = @('$true', '$false')
    'System.Object[]' = @(
        '@()',
        '@(5)',
        ',@(1, 2)',
        '@(@(1, 2), @(3, 4))',
        '@(1, 2)',
        "@('a', 'b')",
        '@(10, 20, 30)'
    )
    'System.Void'    = @('$null')
}

#: The operators the value domain has to answer for. The comparison family is here because a
#: comparison is not a Boolean when the left operand is a collection — it filters and returns an
#: array — and that is a shape the grid must record rather than a rule read off a name.
$BinaryOperators = @(
    '+', '-', '*', '/', '%',
    '-band', '-bor', '-bxor', '-shl', '-shr',
    '-eq', '-ne', '-lt', '-le', '-gt', '-ge'
)

#: The types a cast may target, which is the conversion grid's second axis. `void` is excluded: it
#: is the discard idiom rather than a type a value can have.
$ConversionTargets = @(
    'byte', 'sbyte', 'int16', 'uint16', 'int', 'uint32', 'long', 'uint64',
    'single', 'double', 'decimal', 'string', 'char', 'bool', 'array'
)

function Get-Outcome {
    <#
    .SYNOPSIS
    What a value is, as the grid records it: a type name, or a marker for what happened instead.

    .DESCRIPTION
    `throw` and `null` are distinct answers and neither is a type. A cell that throws is one the
    domain must model as throwing; a cell that yields $null has no type to report, because
    $null.GetType() throws and naming one would put a type into the grid that no value has.
    #>
    param($Value)
    if ($null -eq $Value) { return 'null' }
    return $Value.GetType().FullName
}

function Get-AxisLabel {
    <#
    .SYNOPSIS
    Which row and column of the grid a value belongs on, which is how the witness table is keyed.

    .DESCRIPTION
    An axis and a cell name $null differently, and both names are right for their job. A cell says
    `null` because it has no type to report. An axis has to be labelled with something a witness can
    be filed under, and `System.Void` is the label the grid uses for the shape that holds no value.
    #>
    param($Value)
    $outcome = Get-Outcome $Value
    if ($outcome -eq 'null') { return 'System.Void' }
    return $outcome
}

#: Each operand is evaluated once and each operator compiled once, because compiling a script block
#: per cell is what made a first version of this take longer than the grid is large: the work is
#: tens of thousands of applications, and only sixteen of them are a different program.
#:
#: The operands reach the compiled block through script-scoped variables rather than as parameters.
#: Parameter binding would unroll a collection and re-wrap the value, so `@(1, 2)` would not arrive
#: as an `Object[]` — and the rows this grid exists to settle are exactly the ones with a collection
#: on one side.
#:
#: The result comes back through an assignment the compiled block performs, and is then read out of
#: the variable, because every other way of getting it out changes it. The call operator writes to
#: the pipeline, which unrolls a collection and yields nothing at all for an empty one, so
#: `10, 20, 30 -ne 20` came back as its elements. `InvokeReturnAsIs` does not unroll but still
#: collapses a one-element collection to its element, so `[array] 5` reported an Int32 and a filter
#: matching one thing reported the thing. Assignment is also the shape every ledgered row is written
#: in, so what is captured is what `$t = <expression>` leaves in `$t`.
#:
#: A witness is built the same way and for the same reason. Invoking `@()` for its output gave
#: $null, which filed a value with no type at all under `System.Object[]` and put $null's answers
#: into the collection row of a shipped grid — `@() - 1` is where its Int32 came from.
#:
#: None of the three sites can share a helper that hands the value back, because returning it from a
#: function collapses it exactly as the call operator does. The variable is the carrier, so reading
#: it has to happen where the block was invoked.
#:
#: What a witness turned out to be is then checked against the row that claims it. The row key is
#: the label the grid's axis carries and nothing downstream re-derives it, so a witness that is not
#: what its row says mislabels every cell in it, silently and for good. This is the check that was
#: missing.
$TypeNames = @($Witnesses.Keys)
$Values = [ordered] @{}
foreach ($name in $TypeNames) {
    $collected = [System.Collections.ArrayList]::new()
    foreach ($text in $Witnesses[$name]) {
        $build = [ScriptBlock]::Create("`$script:OperandOut = $text")
        $script:OperandOut = $null
        [void] $build.InvokeReturnAsIs()
        $actual = Get-AxisLabel $script:OperandOut
        if ($actual -ne $name) {
            throw "witness $text is a $actual, but it is listed under $name"
        }
        [void] $collected.Add($script:OperandOut)
    }
    $Values[$name] = $collected
}

$Binary = [ordered] @{}

foreach ($op in $BinaryOperators) {
    Write-Verbose "operator $op"
    $apply = [ScriptBlock]::Create("`$script:OperandOut = `$script:OperandL $op `$script:OperandR")
    $byLeft = [ordered] @{}
    foreach ($left in $TypeNames) {
        $byRight = [ordered] @{}
        foreach ($right in $TypeNames) {
            $seen = [System.Collections.Generic.HashSet[string]]::new()
            foreach ($l in $Values[$left]) {
                foreach ($r in $Values[$right]) {
                    $script:OperandL = $l
                    $script:OperandR = $r
                    try {
                        $script:OperandOut = $null
                        [void] $apply.InvokeReturnAsIs()
                        [void] $seen.Add((Get-Outcome $script:OperandOut))
                    } catch {
                        [void] $seen.Add('throw')
                    }
                }
            }
            $byRight[$right] = @($seen | Sort-Object)
        }
        $byLeft[$left] = $byRight
    }
    $Binary[$op] = $byLeft
}

$Conversions = [ordered] @{}

foreach ($target in $ConversionTargets) {
    Write-Verbose "cast [$target]"
    $convert = [ScriptBlock]::Create("`$script:OperandOut = [$target] `$script:OperandL")
    $bySource = [ordered] @{}
    foreach ($source in $TypeNames) {
        $seen = [System.Collections.Generic.HashSet[string]]::new()
        foreach ($v in $Values[$source]) {
            $script:OperandL = $v
            try {
                $script:OperandOut = $null
                [void] $convert.InvokeReturnAsIs()
                [void] $seen.Add((Get-Outcome $script:OperandOut))
            } catch {
                [void] $seen.Add('throw')
            }
        }
        $bySource[$source] = @($seen | Sort-Object)
    }
    $Conversions[$target] = $bySource
}

$document = [ordered] @{
    schema = [ordered] @{ version = $SchemaVersion }
    host = [ordered] @{
        ps_version = $PSVersionTable.PSVersion.ToString()
        edition = "$($PSVersionTable.PSEdition)"
        culture = [System.Threading.Thread]::CurrentThread.CurrentCulture.Name
        authoritative = (-not $Unauthoritative)
    }
    witnesses = $Witnesses
    binary = $Binary
    conversions = $Conversions
}

if ($Unauthoritative) {
    $OutputDirectory = Join-Path ([System.IO.Path]::GetTempPath()) 'pwsh-operators'
}
if (-not (Test-Path $OutputDirectory)) {
    [void] (New-Item -ItemType Directory -Path $OutputDirectory -Force)
}

$path = Join-Path $OutputDirectory 'pwsh-operators.json'
$json = ($document | ConvertTo-Json -Depth $JsonDepth) -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($path, $json, (New-Object System.Text.UTF8Encoding $false))
Write-Output "wrote $path"
