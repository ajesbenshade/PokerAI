#!/bin/bash

# Quick PokerAI Training Status Check

echo "🎯 PokerAI Training Status"
echo "=========================="

# Training process
if ps aux | grep "python.*train.py" | grep -v grep > /dev/null; then
    echo "✅ Status: RUNNING"
    ps aux | grep "python.*train.py" | grep -v grep | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "Runtime:", $10}'
else
    echo "❌ Status: NOT RUNNING"
fi

# GPU utilization
echo "🎮 GPU Usage:"
rocm-smi --showuse | grep "GPU use" | sed 's/GPU\[0\]/GPU[0] (Primary)/; s/GPU\[1\]/GPU[1] (Secondary)/'

# Progress metrics
if [ -f "training_metrics.csv" ]; then
    lines=$(wc -l < training_metrics.csv)
    if [ $lines -gt 1 ]; then
        echo "📊 Progress: $((lines-1)) training samples"
        tail -2 training_metrics.csv | head -1 | awk -F',' '{print "   Latest: Hand", $1, "Reward:", $2}'
    else
        echo "📊 Progress: Initializing (CFR training phase)"
    fi
fi

# Recent log activity
if [ -f "training_output.log" ]; then
    echo "📝 Last Activity:"
    tail -3 training_output.log | sed 's/^/   /'
fi

echo
echo "💡 Commands:"
echo "   ./training_status.sh      # Quick status check"
echo "   ./training_progress.sh    # Detailed progress & ETA"
echo "   ./monitor_training.sh     # Continuous monitoring (5min intervals)"
echo "   ./monitor_training.sh 60  # Continuous monitoring (1min intervals)"
echo "   tail -f training_output.log  # Live log monitoring"