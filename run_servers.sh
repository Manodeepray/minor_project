cd ~/minor_project || exit

# Start a new tmux session named 'minor_project'
tmux new-session -d -s minor_project

# Run processor.py in the first pane
tmux send-keys -t minor_project "python processor.py" C-m

# Split window horizontally and run server.py
tmux split-window -h
tmux send-keys -t minor_project "python server.py" C-m

# Split window vertically and run ngrok
tmux split-window -v
tmux send-keys -t minor_project "ngrok http 5000" C-m

# Attach to the tmux session
tmux attach -t minor_project