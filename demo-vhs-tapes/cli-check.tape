Output demo-vhs-gifs/cli-check.gif

Set Shell "bash"
Set FontSize 15
Set Width 1200
Set Height 600
Set Theme "Dracula"
Set TypingSpeed 40ms
Set Padding 16

# Hidden: activate venv and pre-warm models
Hide
Type "source .venv/bin/activate && export HF_HUB_OFFLINE=1"
Enter
Type 'longtracer check "a" "b"'
Enter
Sleep 10s
Type "clear"
Enter
Sleep 500ms
Show

# Visible: intro comment
Type "# LongTracer — detect hallucinations in LLM responses"
Enter
Sleep 1s

# The check command
Type 'longtracer check "The Eiffel Tower is 330m tall and in Berlin." "The Eiffel Tower is in Paris, France. It is 330 metres tall."'
Enter

# Wait for output
Sleep 10s
