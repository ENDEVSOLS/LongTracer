Output demo-vhs-gifs/python-api.gif

Set Shell "bash"
Set FontSize 20
Set Width 1200
Set Height 600
Set Theme "Dracula"
Set TypingSpeed 40ms
Set Padding 16

Hide
Type "source .venv/bin/activate"
Enter
# Pre-warm models
Type 'python3 -c "from longtracer import check; check(\"x\",[\"x\"])"'
Enter
Sleep 10s
Type "clear"
Enter
Sleep 500ms
Show

Type "python3"
Enter
Sleep 500ms

Type "from longtracer import check"
Enter
Sleep 1s

Type 'result = check("Python was created by James Gosling in 1991.", ["Python was created by Guido van Rossum and released in 1991."])'
Enter
Sleep 3s

Type "print(result.verdict, result.trust_score)"
Enter
Sleep 1s

Type "print(result.hallucination_count, 'hallucination(s)')"
Enter
Sleep 1s

# Show claim details
Type "for c in result.claims: print('✓' if c['supported'] else '✗', c['claim'][:80])"
Enter
Sleep 3s
