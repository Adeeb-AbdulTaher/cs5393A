"""
Collect more Chart bugs (6-15) to extend the dataset
Uses WSL to run Defects4J commands
"""
import os
import sys
import subprocess
import time

def run_wsl_command(cmd):
    """Run a command in WSL"""
    result = subprocess.run(['wsl', 'bash', '-c', cmd], 
                          capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def collect_chart_bug(bug_id):
    """Collect a single Chart bug"""
    print(f"\n{'='*70}")
    print(f"Collecting Chart-{bug_id}...")
    print(f"{'='*70}")
    
    work_dir = f'/tmp/Chart_{bug_id}'
    
    # Step 1: Checkout
    print(f"  [1/4] Checking out Chart-{bug_id}b...")
    cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && " \
          f"sudo /root/defects4j/framework/bin/defects4j checkout " \
          f"-p Chart -v {bug_id}b -w {work_dir}"
    success, stdout, stderr = run_wsl_command(cmd)
    if not success:
        return False, f"Checkout failed: {stderr[:200]}"
    print(f"    ✓ Checkout successful")
    
    # Step 2: Run tests
    print(f"  [2/4] Running tests...")
    cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && " \
          f"cd {work_dir} && " \
          f"sudo /root/defects4j/framework/bin/defects4j test"
    success, stdout, stderr = run_wsl_command(cmd)
    if not success:
        return False, f"Test run failed: {stderr[:200]}"
    
    # Extract failing test count
    failing_count = stdout.count('Failing tests:')
    print(f"    ✓ Tests completed (found {failing_count} failing test lines)")
    
    # Step 3: Get coverage
    print(f"  [3/4] Getting coverage...")
    cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && " \
          f"cd {work_dir} && " \
          f"sudo /root/defects4j/framework/bin/defects4j coverage"
    success, stdout, stderr = run_wsl_command(cmd)
    if not success:
        return False, f"Coverage failed: {stderr[:200]}"
    print(f"    ✓ Coverage collected")
    
    # Step 4: Copy files to Windows
    print(f"  [4/4] Copying files to Windows...")
    local_dir = f"multi_bug_data/Chart_{bug_id}"
    os.makedirs(local_dir, exist_ok=True)
    
    files_to_copy = ['coverage.xml', 'summary.csv', 'failing_tests', 'all_tests']
    for filename in files_to_copy:
        cmd = f"cp {work_dir}/{filename} /home/adeeb/Chart_{bug_id}_{filename} 2>/dev/null || echo 'File not found'"
        run_wsl_command(cmd)
        
        # Copy from WSL home to Windows
        wsl_path = f"/home/adeeb/Chart_{bug_id}_{filename}"
        if os.path.exists(f"\\\\wsl$\\Ubuntu\\home\\adeeb\\Chart_{bug_id}_{filename}"):
            import shutil
            shutil.copy(f"\\\\wsl$\\Ubuntu\\home\\adeeb\\Chart_{bug_id}_{filename}", 
                       os.path.join(local_dir, filename))
        else:
            # Try direct WSL copy
            cmd = f"wsl bash -c 'if [ -f {work_dir}/{filename} ]; then cp {work_dir}/{filename} {local_dir.replace(chr(92), \"/\")}/{filename}; fi'"
            os.system(cmd)
    
    print(f"    ✓ Files copied to {local_dir}")
    return True, "Success"

print("=" * 70)
print("Collecting More Chart Bugs (6-15)")
print("=" * 70)
print("This will collect 10 more Chart bugs to extend the dataset")
print("=" * 70)

# Collect Chart bugs 6-15
bug_ids = list(range(6, 16))
successful = []
failed = []

for bug_id in bug_ids:
    try:
        success, message = collect_chart_bug(bug_id)
        if success:
            successful.append(bug_id)
            print(f"\n✓ Chart-{bug_id} collected successfully")
        else:
            failed.append((bug_id, message))
            print(f"\n✗ Chart-{bug_id} failed: {message}")
    except Exception as e:
        failed.append((bug_id, str(e)))
        print(f"\n✗ Chart-{bug_id} failed with exception: {e}")
    
    # Small delay between bugs
    time.sleep(2)

print(f"\n{'='*70}")
print("Collection Summary")
print(f"{'='*70}")
print(f"✅ Successful: {len(successful)} bugs - {successful}")
print(f"❌ Failed: {len(failed)} bugs")
if failed:
    for bug_id, error in failed:
        print(f"  - Chart-{bug_id}: {error[:100]}")

print(f"\n✓ Collection complete!")
print(f"\nNext steps:")
print(f"  1. Run: python combine_multi_bug_data.py")
print(f"  2. Run: python train_combined_model.py")
print(f"  3. Generate reports")

