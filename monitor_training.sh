#!/bin/bash

# PokerAI Training Monitor Script
# Provides periodic progress updates for training runs

MONITOR_INTERVAL=${1:-300}  # Default 5 minutes
LOG_FILE="training_output.log"
METRICS_FILE="training_metrics.csv"

echo "🎯 PokerAI Training Monitor Started"
echo "===================================="
echo "Monitoring interval: ${MONITOR_INTERVAL} seconds"
echo "Press Ctrl+C to stop monitoring"
echo

# Function to get training status
get_training_status() {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

    # Check if training is running
    if ps aux | grep "python.*train.py" | grep -v grep > /dev/null; then
        echo "✅ Training: RUNNING"
        ps aux | grep "python.*train.py" | grep -v grep | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
    else
        echo "❌ Training: NOT RUNNING"
        return 1
    fi

    # GPU status
    echo "🎮 GPU Status:"
    rocm-smi --showuse | grep "GPU use" | sed 's/^/   /'

    # Training progress
    if [ -f "$METRICS_FILE" ]; then
        local lines=$(wc -l < "$METRICS_FILE")
        if [ $lines -gt 1 ]; then
            echo "📊 Training Progress: $((lines-1)) data points logged"
            tail -2 "$METRICS_FILE" | head -1 | awk -F',' '{print "   Latest hand:", $1, "Avg reward:", $2}'
        else
            echo "📊 Training Progress: Still initializing (header only)"
        fi
    fi

    # Recent log activity
    if [ -f "$LOG_FILE" ]; then
        echo "📝 Recent Activity:"
        tail -5 "$LOG_FILE" | grep -E "(INFO|Simulation time|CFR|Phase|Training Progress)" | tail -3 | sed 's/^/   /'
    fi

    echo
}

# Main monitoring loop
while true; do
    get_training_status
    sleep $MONITOR_INTERVAL
done