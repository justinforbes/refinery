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

#: Two tables and five cast targets were added to a document that had only `binary` and
#: `conversions`, so a reader written for the old one cannot tell the two apart by looking. The
#: version is this grid's own and moves with this grid: `data.py` checks it against the shared
#: `SCHEMA_VERSION`, which is the metadata tables' number and has no business being this one.
$SchemaVersion = 2
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
#: A number whose type carries no integer width of its own splits on its magnitude, because the
#: operations that need a width take the narrowest one that holds the value: `3000000000.0 -shl 1`
#: is a UInt32 and `5000000000.0 -shl 1` an Int64 where `1.5 -shl 1` is an Int32. Without a witness
#: at each of those magnitudes the row named `Int32` alone, which is a rung and not the rule.
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
    'System.Double'  = @('0.0', '1.5', '-1.5', '3000000000.0', '5000000000.0', '[double]::MaxValue')
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

#: The operators the value domain has to answer for, in the shape where both operands are values.
#: The comparison family is here because a comparison is not a Boolean when the left operand is a
#: collection — it filters and returns an array — and that is a shape the grid must record rather
#: than a rule read off a name. The text family is here for the same reason in reverse: `-split`
#: answers with an array and `-join` with a String whatever it was given, and `-replace` reads a
#: two-element right operand as a pattern and a replacement, so what each produces is a measurement
#: and not a reading of its name.
#:
#: `-and` and `-or` are absent because they short-circuit, so an application of one is not a
#: function of its operands and there is no cell to record. `-is`, `-isnot` and `-as` are absent
#: because their right operand is a type rather than a value, which is a different grid.
$BinaryOperators = @(
    '+', '-', '*', '/', '%',
    '-band', '-bor', '-bxor', '-shl', '-shr',
    '-eq', '-ne', '-lt', '-le', '-gt', '-ge',
    '-xor',
    '-contains', '-notcontains', '-in', '-notin',
    '-like', '-notlike', '-match', '-notmatch',
    '-replace', '-creplace', '-ireplace',
    '-split', '-join'
)

#: The operators that take one operand, which have no cell in a grid over pairs and were left
#: unmeasured by one. `-bnot` is the reason this table exists: the width it complements at is
#: decided by the operand's *value* where the operand has no width of its own, so the set a cell
#: records is what shows that a single width written down is a floor rather than the rule.
$UnaryOperators = @('-', '+', '-not', '-bnot')

#: The types a cast may target, which is the conversion grid's second axis. `void` is excluded: it
#: is the discard idiom rather than a type a value can have. The array targets are here because a
#: cast to one is how a script says what a collection holds, and the answer is a different type from
#: the accelerator that was written — `[byte[]]` builds a `System.Byte[]` where `[array]` builds a
#: `System.Object[]`.
$ConversionTargets = @(
    'byte', 'sbyte', 'int16', 'uint16', 'int', 'uint32', 'long', 'uint64',
    'single', 'double', 'decimal', 'string', 'char', 'bool', 'array',
    'byte[]', 'char[]', 'int[]', 'string[]', 'object[]'
)

#: The operators whose right operand is a type rather than a value, so that their grid is over a
#: value's type and a written target — the conversion grid's axes, not the binary grid's. `-as` is
#: here rather than among the conversions because it answers a failed cast with $null where a cast
#: throws, and which of the two a script wrote is a difference the domain has to keep.
$TypeOperators = @('-is', '-isnot', '-as')

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

function Get-CellOutcomes {
    <#
    .SYNOPSIS
    Every outcome recorded anywhere under a table, however deeply it is keyed.

    .DESCRIPTION
    Handing these back through the pipeline is safe where handing a measured value back is not: what
    a table holds is outcome *names*, and unrolling a list of names loses nothing.
    #>
    param($Node)
    if ($Node -is [System.Collections.IDictionary]) {
        foreach ($key in @($Node.Keys)) { Get-CellOutcomes $Node[$key] }
    } else {
        $Node
    }
}

function Assert-Measured {
    <#
    .SYNOPSIS
    Refuse a table in which some operator threw everywhere, because that is what a broken
    measurement looks like and not what an operator does.

    .DESCRIPTION
    Every cell of an operator is filled from the same compiled block, and the only thing separating
    a throw the operator caused from a throw the harness caused is that the harness causes all of
    them. So an operator with nothing but throws under it is refused rather than shipped: the block
    is compiled once and outside the try that records a throw, which is what makes this the one
    failure the try can still hide.
    #>
    param($Table, [string] $Kind)
    foreach ($op in @($Table.Keys)) {
        $survivors = @(Get-CellOutcomes $Table[$op] | Where-Object { $_ -ne 'throw' })
        if ($survivors.Count -eq 0) {
            throw "every $Kind application of $op threw, so nothing was measured but the harness"
        }
    }
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

$Unary = [ordered] @{}

foreach ($op in $UnaryOperators) {
    Write-Verbose "unary $op"
    $apply = [ScriptBlock]::Create("`$script:OperandOut = $op `$script:OperandL")
    $byOperand = [ordered] @{}
    foreach ($name in $TypeNames) {
        $seen = [System.Collections.Generic.HashSet[string]]::new()
        foreach ($v in $Values[$name]) {
            $script:OperandL = $v
            try {
                $script:OperandOut = $null
                [void] $apply.InvokeReturnAsIs()
                [void] $seen.Add((Get-Outcome $script:OperandOut))
            } catch {
                [void] $seen.Add('throw')
            }
        }
        $byOperand[$name] = @($seen | Sort-Object)
    }
    $Unary[$op] = $byOperand
}

$TypeTests = [ordered] @{}

foreach ($op in $TypeOperators) {
    Write-Verbose "type operator $op"
    $bySource = [ordered] @{}
    foreach ($name in $TypeNames) {
        $bySource[$name] = [ordered] @{}
    }
    foreach ($target in $ConversionTargets) {
        $apply = [ScriptBlock]::Create("`$script:OperandOut = `$script:OperandL $op [$target]")
        foreach ($name in $TypeNames) {
            $seen = [System.Collections.Generic.HashSet[string]]::new()
            foreach ($v in $Values[$name]) {
                $script:OperandL = $v
                try {
                    $script:OperandOut = $null
                    [void] $apply.InvokeReturnAsIs()
                    [void] $seen.Add((Get-Outcome $script:OperandOut))
                } catch {
                    [void] $seen.Add('throw')
                }
            }
            $bySource[$name][$target] = @($seen | Sort-Object)
        }
    }
    $TypeTests[$op] = $bySource
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

Assert-Measured $Binary 'binary'
Assert-Measured $Unary 'unary'
Assert-Measured $TypeTests 'type-operator'
Assert-Measured $Conversions 'cast to'

#: Nine thousand cells hold seventy-two distinct answers between them, and naming an axis in every
#: cell wrote a type name nine thousand times over. So the document lists the answers once, lists
#: each axis once, and every cell becomes a position in both — which is the difference between
#: forty kilobytes and two and a half.
#:
#: The tables are built above under their names and encoded here rather than being filled with
#: positions as they go, because a position is unreadable and `Assert-Measured` has to be able to
#: see what a cell says. Capture in the form that can be checked, encode once it has been.
$Outcomes = [System.Collections.ArrayList]::new()
$OutcomeIndex = @{}

function Get-OutcomeSlot {
    <#
    .SYNOPSIS
    The position of an outcome list in the document's table of them, adding it if it is new.
    #>
    param($Outcome)
    $key = $Outcome -join "`t"
    if (-not $OutcomeIndex.ContainsKey($key)) {
        $OutcomeIndex[$key] = $Outcomes.Count
        [void] $Outcomes.Add(@($Outcome))
    }
    return $OutcomeIndex[$key]
}

function ConvertTo-IndexedRow {
    <#
    .SYNOPSIS
    One axis of a table, as the outcome positions its keys hold, in the axis's own order.

    .DESCRIPTION
    The result is wrapped by the comma operator on the way out for the reason every other value in
    this script travels the way it does: returning a collection from a function collapses it, and a
    row of one would come back as a bare number.
    #>
    param($Row, $Axis)
    $out = [System.Collections.ArrayList]::new()
    foreach ($key in $Axis) { [void] $out.Add((Get-OutcomeSlot $Row[$key])) }
    return ,@($out)
}

function ConvertTo-IndexedTable {
    <#
    .SYNOPSIS
    Two axes of a table, outer first, as rows of outcome positions.
    #>
    param($Table, $Outer, $Inner)
    $out = [System.Collections.ArrayList]::new()
    foreach ($key in $Outer) { [void] $out.Add((ConvertTo-IndexedRow $Table[$key] $Inner)) }
    return ,@($out)
}

$IndexedBinary = [ordered] @{}
foreach ($op in $BinaryOperators) {
    $IndexedBinary[$op] = ConvertTo-IndexedTable $Binary[$op] $TypeNames $TypeNames
}

$IndexedUnary = [ordered] @{}
foreach ($op in $UnaryOperators) {
    $IndexedUnary[$op] = ConvertTo-IndexedRow $Unary[$op] $TypeNames
}

$IndexedTypeTests = [ordered] @{}
foreach ($op in $TypeOperators) {
    $IndexedTypeTests[$op] = ConvertTo-IndexedTable $TypeTests[$op] $TypeNames $ConversionTargets
}

$IndexedConversions = [ordered] @{}
foreach ($target in $ConversionTargets) {
    $IndexedConversions[$target] = ConvertTo-IndexedRow $Conversions[$target] $TypeNames
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
    types = $TypeNames
    targets = $ConversionTargets
    outcomes = @($Outcomes)
    binary = $IndexedBinary
    unary = $IndexedUnary
    type_tests = $IndexedTypeTests
    conversions = $IndexedConversions
}

if ($Unauthoritative) {
    $OutputDirectory = Join-Path ([System.IO.Path]::GetTempPath()) 'pwsh-operators'
}
if (-not (Test-Path $OutputDirectory)) {
    [void] (New-Item -ItemType Directory -Path $OutputDirectory -Force)
}

$path = Join-Path $OutputDirectory 'pwsh-operators.json'
$json = $document | ConvertTo-Json -Depth $JsonDepth -Compress
[System.IO.File]::WriteAllText($path, $json, (New-Object System.Text.UTF8Encoding $false))
Write-Output "wrote $path"
