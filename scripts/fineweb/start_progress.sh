#!/bin/bash
# Run after every workspace restart: bash /root/runs/fineweb4way/start_progress.sh
R=/root/runs/fineweb4way
tmux kill-session -t progress 2>/dev/null
tmux new -d -s progress "while true; do python3 $R/progress_report.py > $R/progress_latest.txt 2>&1; cat $R/progress_latest.txt; echo; cat $R/progress_latest.txt >> $R/progress.log; echo >> $R/progress.log; sleep 3600; done"
tmux kill-session -t git-milestones 2>/dev/null
tmux new -d -s git-milestones "python3 $R/git_milestones.py >> $R/git_milestones.log 2>&1; exec bash"
