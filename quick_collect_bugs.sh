#!/bin/bash
# Quick script to collect Chart bugs 1-5 for addressing class imbalance
# Run this in WSL: bash quick_collect_bugs.sh

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

echo "=========================================="
echo "Collecting Chart Bugs 1-5"
echo "=========================================="

# Create output directory in Windows accessible location
OUTPUT_DIR="/mnt/d/adeeb/Downloads/Shibbir Presentation/project/multi_bug_data"
mkdir -p "$OUTPUT_DIR"

for i in 1 2 3 4 5; do
    echo ""
    echo "----------------------------------------"
    echo "Processing Chart-$i"
    echo "----------------------------------------"
    
    WORK_DIR="/tmp/Chart_${i}b"
    BUG_DIR="$OUTPUT_DIR/Chart_${i}"
    
    # Checkout
    echo "  [1/3] Checking out Chart-$i..."
    sudo /root/defects4j/framework/bin/defects4j checkout -p Chart -v ${i}b -w "$WORK_DIR" 2>&1 | grep -E "(OK|FAIL|Checking out)" || true
    
    if [ ! -f "$WORK_DIR/.defects4j.config" ]; then
        echo "  ✗ Checkout failed, skipping..."
        continue
    fi
    
    # Run tests
    echo "  [2/3] Running tests..."
    cd "$WORK_DIR"
    sudo /root/defects4j/framework/bin/defects4j test 2>&1 | tail -3 || true
    
    # Get coverage
    echo "  [3/3] Getting coverage..."
    sudo /root/defects4j/framework/bin/defects4j coverage 2>&1 | tail -5 || true
    
    # Copy files
    echo "  [4/4] Copying data files..."
    mkdir -p "$BUG_DIR"
    sudo cp "$WORK_DIR/coverage.xml" "$BUG_DIR/" 2>/dev/null || true
    sudo cp "$WORK_DIR/summary.csv" "$BUG_DIR/" 2>/dev/null || true
    sudo cp "$WORK_DIR/failing_tests" "$BUG_DIR/" 2>/dev/null || true
    sudo cp "$WORK_DIR/all_tests" "$BUG_DIR/" 2>/dev/null || true
    sudo chmod 644 "$BUG_DIR"/* 2>/dev/null || true
    
    # Count failing tests
    if [ -f "$BUG_DIR/failing_tests" ]; then
        FAILING_COUNT=$(grep -c "^---" "$BUG_DIR/failing_tests" 2>/dev/null || echo "0")
        echo "  ✓ Chart-$i complete: $FAILING_COUNT failing test(s)"
    else
        echo "  ✓ Chart-$i complete"
    fi
done

echo ""
echo "=========================================="
echo "Collection Complete!"
echo "=========================================="
echo ""
echo "Data saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Run: python combine_multi_bug_data.py"
echo "  2. Retrain model with combined data"
echo ""

