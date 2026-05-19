Output demo-vhs-gifs/tui-demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 800
Set Theme "Dracula"
Set TypingSpeed 40ms
Set Padding 16

Hide
Type "source .venv/bin/activate && export HF_HUB_OFFLINE=1"
Enter
# Pre-warm models so loading spinner is fast
Type "python demos/longtracer_demo.py"
Enter
Sleep 10s
Type "clear"
Enter
Sleep 500ms
Show

Type "# Run the full interactive verification demo"
Enter
Sleep 1s

Type "python demos/longtracer_demo.py"
Enter

# The script has several sleep() calls, taking ~8-10 seconds total.
Sleep 12s
