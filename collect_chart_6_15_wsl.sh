#!/bin/bash
# Collect Chart bugs 6-15 in WSL
# Run this in WSL: bash collect_chart_6_15_wsl.sh

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

echo "============================================================"
echo "Collecting Chart Bugs 6-15"
echo "============================================================"

SUCCESSFUL=()
FAILED=()

for bug_id in {6..15}; do
    echo ""
    echo "============================================================"
    echo "Processing Chart-${bug_id}"
    echo "============================================================"
    
    WORK_DIR="/tmp/Chart_${bug_id}"
    WINDOWS_DIR="/mnt/d/adeeb/Downloads/Shibbir Presentation/project/multi_bug_data/Chart_${bug_id}"
    
    # Create Windows directory
    mkdir -p "$WINDOWS_DIR"
    
    # 1. Checkout
    echo "  [1/4] Checking out Chart-${bug_id}b..."
    sudo /root/defects4j/framework/bin/defects4j checkout -p Chart -v ${bug_id}b -w "$WORK_DIR" 2>&1 | tail -5
    if [ $? -ne 0 ]; then
        FAILED+=("Chart-${bug_id}: Checkout failed")
        continue
    fi
    echo "    ✓ Checkout done"
    
    # 2. Run tests
    echo "  [2/4] Running tests..."
    cd "$WORK_DIR"
    sudo /root/defects4j/framework/bin/defects4j test 2>&1 | tail -10
    if [ $? -ne 0 ]; then
        FAILED+=("Chart-${bug_id}: Tests failed")
        continue
    fi
    FAILING_COUNT=$(grep -c "Failing tests:" <<< "$(sudo /root/defects4j/framework/bin/defects4j test 2>&1)" || echo "0")
    echo "    ✓ Tests done"
    
    # 3. Get coverage
    echo "  [3/4] Getting coverage..."
    sudo /root/defects4j/framework/bin/defects4j coverage 2>&1 | tail -5
    if [ $? -ne 0 ]; then
        FAILED+=("Chart-${bug_id}: Coverage failed")
        continue
    fi
    echo "    ✓ Coverage done"
    
    # 4. Copy files
    echo "  [4/4] Copying files..."
    for file in coverage.xml summary.csv failing_tests all_tests; do
        if [ -f "$WORK_DIR/$file" ]; then
            cp "$WORK_DIR/$file" "$WINDOWS_DIR/$file"
        fi
    done
    echo "    ✓ Files copied to $WINDOWS_DIR"
    
    SUCCESSFUL+=("Chart-${bug_id}")
    echo ""
    echo "✓ Chart-${bug_id} collected successfully"
done

echo ""
echo "============================================================"
echo "Collection Summary"
echo "============================================================"
echo "✅ Successful (${#SUCCESSFUL[@]}): ${SUCCESSFUL[*]}"
echo "❌ Failed (${#FAILED[@]}):"
for fail in "${FAILED[@]}"; do
    echo "  - $fail"
done

echo ""
echo "✓ Collection complete!"
echo "Next step: Run combine_multi_bug_data.py in Windows"

