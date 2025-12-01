"""
Collect Chart bugs 6-15 using direct WSL commands
"""
import subprocess
import os
import time

def run_wsl(cmd):
    """Run command in WSL"""
    result = subprocess.run(['wsl', 'bash', '-c', cmd], capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

print("=" * 70)
print("Collecting Chart Bugs 6-15")
print("=" * 70)

bug_ids = list(range(6, 16))
successful = []
failed = []

for bug_id in bug_ids:
    print(f"\n{'='*70}")
    print(f"Processing Chart-{bug_id}")
    print(f"{'='*70}")
    
    work_dir = f'/tmp/Chart_{bug_id}'
    local_dir = f'multi_bug_data/Chart_{bug_id}'
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # 1. Checkout
        print(f"  [1/4] Checking out...")
        cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && sudo /root/defects4j/framework/bin/defects4j checkout -p Chart -v {bug_id}b -w {work_dir}"
        success, out, err = run_wsl(cmd)
        if not success:
            failed.append((bug_id, f"Checkout: {err[:100]}"))
            continue
        print(f"    ✓ Checkout done")
        
        # 2. Run tests
        print(f"  [2/4] Running tests...")
        cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && cd {work_dir} && sudo /root/defects4j/framework/bin/defects4j test"
        success, out, err = run_wsl(cmd)
        if not success:
            failed.append((bug_id, f"Tests: {err[:100]}"))
            continue
        failing = out.count('Failing tests:')
        print(f"    ✓ Tests done ({failing} failing test sections)")
        
        # 3. Get coverage
        print(f"  [3/4] Getting coverage...")
        cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && cd {work_dir} && sudo /root/defects4j/framework/bin/defects4j coverage"
        success, out, err = run_wsl(cmd)
        if not success:
            failed.append((bug_id, f"Coverage: {err[:100]}"))
            continue
        print(f"    ✓ Coverage done")
        
        # 4. Copy files
        print(f"  [4/4] Copying files...")
        files = ['coverage.xml', 'summary.csv', 'failing_tests', 'all_tests']
        for fname in files:
            # Copy via WSL to Windows accessible location
            cmd = f"cp {work_dir}/{fname} /mnt/d/adeeb/Downloads/Shibbir\\ Presentation/project/{local_dir}/{fname} 2>&1"
            run_wsl(cmd)
        
        # Verify files exist
        all_exist = all(os.path.exists(os.path.join(local_dir, f)) for f in files)
        if all_exist:
            successful.append(bug_id)
            print(f"    ✓ Files copied")
        else:
            failed.append((bug_id, "Files not copied correctly"))
            
    except Exception as e:
        failed.append((bug_id, str(e)))
        print(f"    ✗ Error: {e}")
    
    time.sleep(1)

print(f"\n{'='*70}")
print("Collection Summary")
print(f"{'='*70}")
print(f"✅ Successful: {len(successful)} bugs - {successful}")
print(f"❌ Failed: {len(failed)} bugs")
if failed:
    for bid, err in failed:
        print(f"  - Chart-{bid}: {err}")

print(f"\n✓ Done! Next: Run combine_multi_bug_data.py")

