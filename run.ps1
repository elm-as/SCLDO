$ErrorActionPreference = "Stop"
$PythonExe = "C:\Users\elmas\AppData\Local\Programs\Python\Python310\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python 3.10 executable not found at: $PythonExe"
    exit 1
}
& $PythonExe -u main.py @args
