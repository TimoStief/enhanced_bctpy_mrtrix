#!/bin/bash
# Monitor the running analysis

echo "=========================================="
echo "Analysis Progress Monitor"
echo "=========================================="

while true; do
    echo ""
    echo "Timestamp: $(date)"
    echo ""
    
    # Check if process is running
    if pgrep -f "complete_metrics_umap_trajectories.py" > /dev/null; then
        echo "✅ Process is running"
        
        # Show process details
        ps aux | grep complete_metrics_umap_trajectories.py | grep -v grep | awk '{print "CPU: "$3"% | Memory: "$6" KB"}'
        
        # Show log tail
        echo ""
        echo "--- Log Output (last 20 lines) ---"
        tail -20 /tmp/complete_analysis.log 2>/dev/null || echo "Log not yet available"
        
        # Check output files
        echo ""
        echo "--- Output Files ---"
        if [ -d "/data/local/129_PK01/derivatives/bct/comprehensive_analysis/" ]; then
            ls -lh /data/local/129_PK01/derivatives/bct/comprehensive_analysis/ | tail -10
        else
            echo "Output directory not yet created"
        fi
    else
        echo "❌ Process finished!"
        echo ""
        echo "Final log output:"
        tail -50 /tmp/complete_analysis.log 2>/dev/null
        
        echo ""
        echo "Final output files:"
        ls -lh /data/local/129_PK01/derivatives/bct/comprehensive_analysis/
        break
    fi
    
    echo ""
    echo "Waiting 60 seconds before next check..."
    sleep 60
done
