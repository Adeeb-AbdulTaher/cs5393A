#!/bin/bash
# Defects4J Helper Script
# This script helps run Defects4J commands with proper environment setup

# Set Java home
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Defects4J path
DEFECTS4J="/root/defects4j/framework/bin/defects4j"

# Check if running as root or with sudo access
if [ "$EUID" -eq 0 ]; then
    # Running as root, use directly
    D4J_CMD="$DEFECTS4J"
elif sudo -n true 2>/dev/null; then
    # Passwordless sudo available
    D4J_CMD="sudo $DEFECTS4J"
else
    # Need sudo with password
    echo "Note: You may be prompted for your sudo password"
    D4J_CMD="sudo $DEFECTS4J"
fi

# Function to checkout a bug
checkout_bug() {
    local project=$1
    local version=$2
    local work_dir=$3
    
    echo "Checking out $project version $version to $work_dir..."
    $D4J_CMD checkout -p $project -v $version -w $work_dir
}

# Function to run tests
run_tests() {
    local work_dir=$1
    
    if [ ! -d "$work_dir" ]; then
        echo "Error: Directory $work_dir does not exist!"
        exit 1
    fi
    
    echo "Running tests in $work_dir..."
    cd "$work_dir"
    $D4J_CMD test
}

# Main execution
case "$1" in
    checkout)
        checkout_bug "$2" "$3" "$4"
        ;;
    test)
        run_tests "$2"
        ;;
    *)
        echo "Usage: $0 {checkout|test} [args]"
        echo ""
        echo "Examples:"
        echo "  $0 checkout Chart 1b /tmp/Chart_1b"
        echo "  $0 test /tmp/Chart_1b"
        exit 1
        ;;
esac

