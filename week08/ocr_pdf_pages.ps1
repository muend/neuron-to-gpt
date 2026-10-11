param(
    [Parameter(Mandatory=$true)][string]$ImageDirectory,
    [Parameter(Mandatory=$true)][string]$Prefix,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
# Windows PowerShell 5.1 exposes the installed Windows Turkish OCR engine.
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime]
$taskAsTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetGenericArguments().Count -eq 1 -and
    $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
function Wait-TaskOperation($Operation, $ResultType) {
    $taskAwaitable = $taskAsTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $taskAwaitable.GetAwaiter().GetResult()
}
$taskLanguage = New-Object Windows.Globalization.Language('tr')
$taskEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($taskLanguage)
if ($null -eq $taskEngine) { throw 'Install the Windows Turkish OCR language before continuing' }
$taskImages = Get-ChildItem -LiteralPath $ImageDirectory -File | Where-Object {
    $_.Name.StartsWith($Prefix + '_') -and $_.Extension -eq '.png'
} | Sort-Object Name
if ($taskImages.Count -eq 0) { throw 'No matching rendered PDF pages' }
$taskPages = @()
foreach ($taskImage in $taskImages) {
    $taskFile = Wait-TaskOperation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($taskImage.FullName)) ([Windows.Storage.StorageFile])
    $taskStream = Wait-TaskOperation ($taskFile.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $taskDecoder = Wait-TaskOperation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($taskStream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $taskBitmap = Wait-TaskOperation ($taskDecoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $taskResult = Wait-TaskOperation ($taskEngine.RecognizeAsync($taskBitmap)) ([Windows.Media.Ocr.OcrResult])
    $taskNumber = [int]([System.IO.Path]::GetFileNameWithoutExtension($taskImage.Name).Substring($Prefix.Length + 1))
    $taskText = ($taskResult.Lines | ForEach-Object { $_.Text }) -join "`n"
    $taskLines = @($taskResult.Lines | ForEach-Object {
        $taskWords = @($_.Words)
        [ordered]@{text=$_.Text; top=($taskWords | ForEach-Object {$_.BoundingRect.Y} | Measure-Object -Minimum).Minimum;
          bottom=($taskWords | ForEach-Object {$_.BoundingRect.Y + $_.BoundingRect.Height} | Measure-Object -Maximum).Maximum}
    })
    $taskPages += [ordered]@{page=$taskNumber; text=$taskText; lines=$taskLines; width=$taskBitmap.PixelWidth; height=$taskBitmap.PixelHeight}
    $taskBitmap.Dispose()
    $taskStream.Dispose()
    Write-Output ('OCR completed page ' + $taskNumber)
}
$taskJson = [ordered]@{engine='Windows OCR'; language='tr'; status='unreviewed_extraction_candidate'; pages=$taskPages} | ConvertTo-Json -Depth 8
if (Test-Path -LiteralPath $OutputPath) { throw 'OCR output already exists; preserve it or select a new output path' }
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputPath), $taskJson, (New-Object System.Text.UTF8Encoding($false)))
