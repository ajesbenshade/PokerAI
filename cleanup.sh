#!/bin/bash
# PokerAI Project Cleanup Script
# Safely removes unnecessary files while preserving important data

echo "🧹 PokerAI Project Cleanup Script"
echo "=================================="

# Function to get user confirmation
confirm() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

echo "Current disk usage:"
du -sh . --exclude=ROCm
echo

echo "Files that will be safely removed:"
echo "✓ Python cache files (__pycache__/, *.pyc) - ALREADY REMOVED"
echo "✓ pytest cache (.pytest_cache/) - ALREADY REMOVED"
echo

# Option 1: Clean up Python cache files
echo "Option 1: Clean up Python cache files"
echo "Files: __pycache__/ directories and *.pyc files"
if confirm "Clean up Python cache files?"; then
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -name "*.pyc" -delete 2>/dev/null
    find . -name "*.pyo" -delete 2>/dev/null
    echo "✅ Cleaned up Python cache files"
else
    echo "⏭️  Keeping Python cache files"
fi

echo

# Option 2: Old TensorBoard runs (keep recent ones)
echo "Option 2: Clean up old TensorBoard runs"
echo "Directory: runs/ ($(du -sh runs/ 2>/dev/null | cut -f1) total)"
echo "Will keep 5 most recent runs, remove older ones"
if confirm "Clean up old TensorBoard runs?"; then
    # Count total runs
    total_runs=$(ls runs/ 2>/dev/null | wc -l)
    if [ $total_runs -gt 5 ]; then
        echo "Keeping 5 most recent runs, removing $(($total_runs - 5)) old ones..."
        ls runs/ | head -n $(($total_runs - 5)) | xargs -I {} rm -rf "runs/{}" 2>/dev/null
        echo "✅ Removed old TensorBoard runs"
    else
        echo "✅ Only $total_runs runs found, keeping all"
    fi
else
    echo "⏭️  Keeping all TensorBoard runs"
fi

echo

# Option 3: Test files cleanup
echo "Option 3: Remove test files"
echo "Files: test_*.py files (development/testing files)"
test_files=$(ls test_*.py 2>/dev/null)
if [ ! -z "$test_files" ]; then
    echo "Found test files:"
    echo "$test_files" | sed 's/^/  • /'
    if confirm "Remove test files?"; then
        rm -f test_*.py
        echo "✅ Removed test files"
    else
        echo "⏭️  Keeping test files"
    fi
else
    echo "✅ No test files found"
fi

echo

# Option 4: Demo files cleanup
echo "Option 4: Remove demo files"
echo "Files: demo_*.py and phase3_demo.py files"
demo_files=$(ls demo_*.py phase3_demo.py 2>/dev/null)
if [ ! -z "$demo_files" ]; then
    echo "Found demo files:"
    echo "$demo_files" | sed 's/^/  • /'
    if confirm "Remove demo files?"; then
        rm -f demo_*.py phase3_demo.py
        echo "✅ Removed demo files"
    else
        echo "⏭️  Keeping demo files"
    fi
else
    echo "✅ No demo files found"
fi

echo

# Option 5: Checkpoint files (DANGER!)
echo "Option 5: Remove checkpoint files (⚠️  DANGER!)"
if ls checkpoint_player_*.pth 1> /dev/null 2>&1; then
    echo "Files: checkpoint_player_*.pth ($(du -sh checkpoint_player_*.pth 2>/dev/null | cut -f1) total)"
    echo "⚠️  WARNING: These contain trained model weights!"
    echo "⚠️  Removing these will delete your trained AI models!"
    if confirm "Remove ALL checkpoint files? (NOT recommended)"; then
        rm -f checkpoint_player_*.pth
        echo "🗑️  Removed checkpoint files"
    else
        echo "✅ Keeping checkpoint files"
    fi
else
    echo "✅ No checkpoint files found"
fi

echo

# Option 6: Experience buffer cleanup
echo "Option 6: Remove experience buffer"
if [ -d "per_buffer.db/" ]; then
    echo "Directory: per_buffer.db/ ($(du -sh per_buffer.db/ 2>/dev/null | cut -f1) total)"
    echo "⚠️  WARNING: This contains training experience data!"
    if confirm "Remove experience buffer?"; then
        rm -rf per_buffer.db/
        echo "🗑️  Removed experience buffer"
    else
        echo "✅ Keeping experience buffer"
    fi
else
    echo "✅ No experience buffer found"
fi

echo

# Option 7: Training metrics cleanup
echo "Option 7: Remove training metrics"
if [ -f "training_metrics.csv" ]; then
    echo "File: training_metrics.csv ($(ls -lh training_metrics.csv | awk '{print $5}') total)"
    if confirm "Remove training metrics?"; then
        rm -f training_metrics.csv
        echo "🗑️  Removed training metrics"
    else
        echo "✅ Keeping training metrics"
    fi
else
    echo "✅ No training metrics found"
fi

echo

# Option 8: Best model files cleanup
echo "Option 8: Remove best model files (⚠️  CAUTION!)"
if ls best_model_player_*.pth 1> /dev/null 2>&1; then
    echo "Files: best_model_player_*.pth ($(du -sh best_model_player_*.pth 2>/dev/null | cut -f1) total)"
    echo "⚠️  CAUTION: These contain your best trained model weights!"
    echo "💡 Consider backing up important models before removing"
    if confirm "Remove ALL best model files?"; then
        rm -f best_model_player_*.pth
        echo "🗑️  Removed best model files"
    else
        echo "✅ Keeping best model files"
    fi
else
    echo "✅ No best model files found"
fi

echo

# Show what remains and final size
echo "Cleanup complete! Summary:"
echo "=========================="
echo "📁 Important files preserved:"
echo "  • Core source: config.py, game.py, models.py, train.py, utils.py"
echo "  • GTO modules: gto.py, enhanced_gto.py, equity_model.py"
echo "  • RL module: rl.py"
echo "  • Scripts: run_train.sh, cleanup.sh"
echo "  • Virtual environment: ROCm/"
echo "  • Current checkpoints: checkpoint_player_*.pth (if kept)"
echo "  • Best models: best_model_player_*.pth (if kept)"
echo

echo "Final disk usage:"
du -sh . --exclude=ROCm
echo

echo "🎉 Cleanup complete!"
echo "💡 Tip: Run 'python -c \"import torch; print(torch.cuda.is_available())\"' to verify PyTorch works"
