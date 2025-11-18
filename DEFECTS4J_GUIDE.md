# Defects4J Usage Guide

## Quick Start

### Step 1: Checkout a Buggy Version

Run this command in your WSL terminal:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
sudo /root/defects4j/framework/bin/defects4j checkout -p Chart -v 1b -w /tmp/Chart_1b
```

**Note:** You'll be prompted for your sudo password. This is needed because Defects4J was installed in `/root`.

### Step 2: Run the Test Suite

After checkout completes, run tests from the checked-out directory:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
cd /tmp/Chart_1b
sudo /root/defects4j/framework/bin/defects4j test
```

## Alternative: Using the Helper Script

1. Copy `run_defects4j.sh` to your WSL home directory:
   ```bash
   cp /mnt/d/adeeb/Downloads/Shibbir\ Presentation/project/run_defects4j.sh ~/
   chmod +x ~/run_defects4j.sh
   ```

2. Use it to checkout:
   ```bash
   ~/run_defects4j.sh checkout Chart 1b /tmp/Chart_1b
   ```

3. Use it to run tests:
   ```bash
   ~/run_defects4j.sh test /tmp/Chart_1b
   ```

## Where to Run

**Inside WSL/Ubuntu terminal** - Open WSL by:
- Typing `wsl` in PowerShell, OR
- Opening "Ubuntu" from Windows Start menu

## Understanding the Output

After running tests, you'll see:
- **Test results summary** showing passed/failed tests
- **Test output files** in the checked-out directory
- **Coverage data** (if coverage tools are configured)

## Next Steps

After running tests, you can:
1. Extract code coverage using JaCoCo or Defects4J's built-in coverage tool
2. Analyze test results
3. Check out the fixed version (`1f` instead of `1b`) for comparison

## Troubleshooting

### Permission Denied
If you get permission errors, make sure to use `sudo` with the commands.

### Java Not Found
Ensure Java 11 is set:
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
java -version  # Should show Java 11
```

### Directory Not Found
Make sure the checkout completed successfully before running tests.

