"""
Collect data from multiple Defects4J bugs to address class imbalance
Automates the process of checking out bugs, running tests, and collecting coverage
"""
import subprocess
import os
import sys

def run_wsl_command(cmd):
    """Run a command in WSL"""
    result = subprocess.run(['wsl', 'bash', '-c', cmd], 
                          capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def checkout_bug(project, bug_id, version='b', work_dir=None):
    """Checkout a bug from Defects4J"""
    if work_dir is None:
        work_dir = f"/tmp/{project}_{bug_id}{version}"
    
    cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && " \
          f"sudo /root/defects4j/framework/bin/defects4j checkout " \
          f"-p {project} -v {bug_id}{version} -w {work_dir}"
    
    print(f"  Checking out {project}-{bug_id}{version}...")
    success, stdout, stderr = run_wsl_command(cmd)
    
    if success:
        print(f"    ✓ Success")
        return work_dir
    else:
        print(f"    ✗ Failed: {stderr[:100]}")
        return None

def run_tests(work_dir):
    """Run tests and collect results"""
    cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && " \
          f"cd {work_dir} && " \
          f"sudo /root/defects4j/framework/bin/defects4j test"
    
    print(f"  Running tests...")
    success, stdout, stderr = run_wsl_command(cmd)
    
    if success:
        # Extract failing test count
        failing_count = stdout.count('Failing tests:')
        print(f"    ✓ Tests completed")
        return True
    else:
        print(f"    ✗ Test run failed")
        return False

def get_coverage(work_dir):
    """Get coverage data"""
    cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && " \
          f"cd {work_dir} && " \
          f"sudo /root/defects4j/framework/bin/defects4j coverage"
    
    print(f"  Getting coverage...")
    success, stdout, stderr = run_wsl_command(cmd)
    
    if success:
        print(f"    ✓ Coverage collected")
        return True
    else:
        print(f"    ✗ Coverage failed")
        return False

def copy_data_files(work_dir, output_dir, bug_id):
    """Copy data files from WSL to Windows"""
    files_to_copy = ['coverage.xml', 'summary.csv', 'failing_tests', 'all_tests']
    
    local_dir = os.path.join(output_dir, f"{bug_id}")
    os.makedirs(local_dir, exist_ok=True)
    
    for file in files_to_copy:
        wsl_path = f"{work_dir}/{file}"
        local_path = os.path.join(local_dir, file)
        
        # Use WSL to read and write
        cmd = f"cat {wsl_path} > /mnt/d/adeeb/Downloads/Shibbir\\ Presentation/project/{bug_id}/{file}"
        run_wsl_command(cmd)
    
    print(f"    ✓ Files copied to {local_dir}")

def get_available_bugs(project):
    """Get list of available bugs for a project"""
    cmd = f"export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && " \
          f"sudo /root/defects4j/framework/bin/defects4j info -p {project}"
    
    success, stdout, stderr = run_wsl_command(cmd)
    
    if success:
        # Parse output to get bug count
        # Defects4J info shows "Number of bugs: X"
        lines = stdout.split('\n')
        for line in lines:
            if 'Number of bugs:' in line:
                try:
                    count = int(line.split(':')[-1].strip())
                    return list(range(1, count + 1))
                except:
                    pass
    
    # Default: return common bug counts
    bug_counts = {
        'Chart': 26,
        'Math': 106,
        'Time': 26,
        'Closure': 174,
        'Lang': 61,
        'Mockito': 38
    }
    return list(range(1, bug_counts.get(project, 10) + 1))

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 60)
print("Multi-Bug Data Collection Script")
print("=" * 60)

# Configuration
projects_to_collect = {
    'Chart': [1, 2, 3, 4, 5],  # Start with first 5 Chart bugs
    # 'Math': [1, 2, 3],        # Uncomment to add more projects
    # 'Time': [1, 2, 3],
}

output_dir = "multi_bug_data"

print(f"\nWill collect bugs from: {list(projects_to_collect.keys())}")
print(f"Output directory: {output_dir}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

all_bugs_data = []

for project, bug_ids in projects_to_collect.items():
    print(f"\n{'='*60}")
    print(f"Processing {project} project")
    print(f"{'='*60}")
    
    for bug_id in bug_ids:
        print(f"\n[{project}-{bug_id}]")
        
        # Checkout buggy version
        work_dir = checkout_bug(project, bug_id, 'b')
        if work_dir is None:
            continue
        
        # Run tests
        if not run_tests(work_dir):
            continue
        
        # Get coverage
        if not get_coverage(work_dir):
            continue
        
        # Copy data files
        copy_data_files(work_dir, output_dir, f"{project}_{bug_id}")
        
        # Read failing tests count
        failing_file = os.path.join(output_dir, f"{project}_{bug_id}", "failing_tests")
        if os.path.exists(failing_file):
            with open(failing_file, 'r') as f:
                content = f.read()
                failing_count = content.count('---')
                all_bugs_data.append({
                    'project': project,
                    'bug_id': bug_id,
                    'failing_tests': failing_count,
                    'data_dir': os.path.join(output_dir, f"{project}_{bug_id}")
                })
                print(f"    Found {failing_count} failing test(s)")

print(f"\n{'='*60}")
print("Collection Summary")
print(f"{'='*60}")

if all_bugs_data:
    total_failing = sum(b['failing_tests'] for b in all_bugs_data)
    print(f"\nCollected {len(all_bugs_data)} bugs")
    print(f"Total failing tests: {total_failing}")
    print(f"\nBugs collected:")
    for bug in all_bugs_data:
        print(f"  - {bug['project']}-{bug['bug_id']}: {bug['failing_tests']} failing tests")
    
    # Save summary
    import json
    with open(os.path.join(output_dir, 'collection_summary.json'), 'w') as f:
        json.dump(all_bugs_data, f, indent=2)
    print(f"\n✓ Summary saved to {output_dir}/collection_summary.json")
    
    print(f"\nNext step: Run combine_multi_bug_data.py to merge all coverage matrices")
else:
    print("No bugs collected successfully")

