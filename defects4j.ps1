# Defects4J WSL Wrapper Script
# Usage: .\defects4j.ps1 [defects4j arguments]
# Example: .\defects4j.ps1 checkout -p Chart -v 1b -w Chart_1b

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Convert Windows path to WSL path if needed for -w argument
$wslArgs = @()
for ($i = 0; $i -lt $Arguments.Length; $i++) {
    if ($Arguments[$i] -eq "-w" -and $i + 1 -lt $Arguments.Length) {
        $wslPath = $Arguments[$i + 1]
        # Convert Windows path to WSL path
        if ($wslPath -match "^[A-Z]:") {
            $drive = $wslPath.Substring(0, 1).ToLower()
            $path = $wslPath.Substring(2).Replace("\", "/")
            $wslPath = "/mnt/$drive$path"
        }
        $wslArgs += "-w"
        $wslArgs += $wslPath
        $i++
    } else {
        $wslArgs += $Arguments[$i]
    }
}

# Run defects4j through WSL with proper environment
$defects4jCmd = "/root/defects4j/framework/bin/defects4j"
$allArgs = $wslArgs -join " "
# Set JAVA_HOME and attempt to work around Perl module issues
wsl bash -c "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && export PATH=\$JAVA_HOME/bin:\$PATH && $defects4jCmd $allArgs"

