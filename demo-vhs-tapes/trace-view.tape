Output demo-vhs-gifs/trace-view.gif

Set Shell "bash"
Set FontSize 15
Set Width 1200
Set Height 600
Set Theme "Dracula"
Set TypingSpeed 40ms
Set Padding 16

Hide
# Pre-run a check to populate trace store
Type "source .venv/bin/activate && export HF_HUB_OFFLINE=1"
Enter
# Pre-run a check to populate trace store
Type 'longtracer check "The Eiffel Tower is in Berlin." "The Eiffel Tower is in Paris."'
Enter
Sleep 10s
Type "clear"
Enter
Sleep 500ms
Show

Type "# View the most recent verification trace"
Enter
Sleep 1s

Type "longtracer view --last"
Enter
Sleep 5s
