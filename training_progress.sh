#!/bin/bash

# PokerAI Progress Estimator

TOTAL_HANDS=200000
PHASE_1_HANDS=10000

# Get current progress
if [ -f "training_metrics.csv" ]; then
    lines=$(wc -l < training_metrics.csv)
    current_hands=$((lines-1))  # Subtract header
else
    current_hands=0
fi

# Get training start time
if ps aux | grep "python.*train.py" | grep -v grep > /dev/null; then
    start_time=$(ps -o lstart= -C python | grep train.py | head -1)
    if [ ! -z "$start_time" ]; then
        start_seconds=$(date -d "$start_time" +%s)
        current_seconds=$(date +%s)
        elapsed_seconds=$((current_seconds - start_seconds))
        elapsed_hours=$((elapsed_seconds / 3600))
        elapsed_minutes=$(( (elapsed_seconds % 3600) / 60 ))
    fi
fi

# Calculate progress
if [ $current_hands -gt 0 ]; then
    progress_percent=$((current_hands * 100 / TOTAL_HANDS))
    hands_per_second=$(echo "scale=2; $current_hands / $elapsed_seconds" | bc 2>/dev/null || echo "0")
    if [ $(echo "$hands_per_second > 0" | bc 2>/dev/null) -eq 1 ]; then
        remaining_hands=$((TOTAL_HANDS - current_hands))
        eta_seconds=$(echo "scale=0; $remaining_hands / $hands_per_second" | bc 2>/dev/null || echo "0")
        eta_hours=$((eta_seconds / 3600))
        eta_minutes=$(( (eta_seconds % 3600) / 60 ))
    fi
fi

# Display progress
echo "🎯 PokerAI Training Progress"
echo "============================"
echo "Target: $TOTAL_HANDS hands"
echo "Current: $current_hands hands"
echo "Progress: ${progress_percent:-0}%"

if [ ! -z "$elapsed_hours" ]; then
    echo "Elapsed: ${elapsed_hours}h ${elapsed_minutes}m"
fi

if [ ! -z "$eta_hours" ]; then
    echo "ETA: ${eta_hours}h ${eta_minutes}m remaining"
fi

# Phase information
if [ $current_hands -lt $PHASE_1_HANDS ]; then
    echo "Phase: 1/2 (CFR Training)"
    echo "Phase Progress: $((current_hands * 100 / PHASE_1_HANDS))% of Phase 1"
else
    phase2_progress=$(( (current_hands - PHASE_1_HANDS) * 100 / (TOTAL_HANDS - PHASE_1_HANDS) ))
    echo "Phase: 2/2 (RL-CFR Hybrid)"
    echo "Phase Progress: ${phase2_progress}% of Phase 2"
fi

echo
echo "💡 CFR training (Phase 1) is slower - expect faster progress in Phase 2"