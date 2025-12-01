#!/bin/bash

# Context Window Impact Analysis Lab - Convenience Runner
# This script makes it easy to run the lab with a single command

set -e  # Exit on error

echo "================================================================================"
echo "Context Window Impact Analysis Lab"
echo "================================================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.x"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Function to check dependencies
check_dependencies() {
    echo "Checking dependencies..."
    python3 -c "import numpy, pandas" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Core dependencies installed"
        return 0
    else
        echo "⚠️  Core dependencies not found"
        return 1
    fi
}

# Function to install dependencies
install_dependencies() {
    echo ""
    echo "Installing dependencies..."
    pip install numpy pandas matplotlib seaborn
    echo "✅ Dependencies installed"
}

# Function to run demo
run_demo() {
    echo ""
    echo "Running quick demo..."
    echo "--------------------------------------------------------------------------------"
    python3 demo.py
}

# Function to run full experiments
run_full() {
    echo ""
    echo "Running full experiments..."
    echo "--------------------------------------------------------------------------------"
    python3 context_lab.py "$@"
}

# Function to generate plots
generate_plots() {
    echo ""
    echo "Generating visualizations..."
    echo "--------------------------------------------------------------------------------"
    python3 visualize.py
    echo ""
    echo "✅ Plots saved to: plots/"
}

# Parse command line arguments
case "${1:-demo}" in
    demo)
        echo "Mode: Quick Demo"
        check_dependencies || install_dependencies
        run_demo
        ;;
    
    full)
        echo "Mode: Full Experiments"
        check_dependencies || install_dependencies
        run_full "${@:2}"
        echo ""
        echo "Next step: Run './run_lab.sh plots' to generate visualizations"
        ;;
    
    plots)
        echo "Mode: Generate Plots"
        check_dependencies || install_dependencies
        if [ ! -f "context_lab_results.json" ]; then
            echo "⚠️  Warning: context_lab_results.json not found"
            echo "   Running experiments first..."
            run_full
        fi
        generate_plots
        ;;
    
    all)
        echo "Mode: Full Lab (Experiments + Plots)"
        check_dependencies || install_dependencies
        run_full "${@:2}"
        generate_plots
        ;;
    
    install)
        echo "Mode: Install Dependencies Only"
        install_dependencies
        echo ""
        echo "✅ Installation complete!"
        echo "   Run './run_lab.sh demo' to test"
        ;;
    
    help|--help|-h)
        echo "Usage: ./run_lab.sh [mode] [options]"
        echo ""
        echo "Modes:"
        echo "  demo          Quick demonstration (default)"
        echo "  full          Run all experiments"
        echo "  plots         Generate visualizations from results"
        echo "  all           Run experiments and generate plots"
        echo "  install       Install dependencies only"
        echo "  help          Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_lab.sh                    # Quick demo"
        echo "  ./run_lab.sh full               # Full experiments"
        echo "  ./run_lab.sh full --experiment 1  # Run experiment 1 only"
        echo "  ./run_lab.sh plots              # Generate plots"
        echo "  ./run_lab.sh all                # Everything"
        echo ""
        echo "For more information, see README.md or QUICK_START.md"
        ;;
    
    *)
        echo "❌ Unknown mode: $1"
        echo "   Run './run_lab.sh help' for usage"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "Complete! 🎉"
echo "================================================================================"

