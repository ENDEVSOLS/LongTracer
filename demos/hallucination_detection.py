"""
LongTracer TUI Demo Script.

This interactive demo showcases the LongTracer hallucination detection
capabilities using the rich terminal library for a polished UI.

Usage:
    python demos/longtracer_demo.py
"""

import sys
import time
from pathlib import Path

# Add repo root to path so we can import longtracer directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner
from rich.align import Align
from rich.layout import Layout
from rich import box

from longtracer import CitationVerifier, check, check_batch


def create_header():
    """Create the header panel."""
    return Panel(
        Align.center(
            "[bold cyan]LongTracer RAG Verification Demo[/bold cyan]\n"
            "[dim]Detect hallucinations in LLM responses using STS + NLI[/dim]"
        ),
        box=box.DOUBLE,
        style="white",
    )


def create_config_panel():
    """Create the configuration display panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("STS Model", "all-MiniLM-L6-v2")
    table.add_row("NLI Model", "nli-deberta-v3-xsmall")
    table.add_row("Threshold", "0.5")
    table.add_row("Mode", "Parallel Batch Verification")
    
    return Panel(table, title="⚙️ Configuration", border_style="cyan")


def format_claim_table(result):
    """Format the claims breakdown table from a verification result."""
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Claim", style="white")
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Note", style="red", width=20)
    
    for c in result.claims:
        supported = c.get("supported", False)
        status = "[green]✅ PASS[/green]" if supported else "[red]❌ FAIL[/red]"
        score = c.get("score", 0.0)
        score_str = f"{score:.2f}"
        
        note = ""
        if c.get("is_hallucination"):
            note = "🚨 Hallucination"
        
        table.add_row(status, c.get("claim", ""), score_str, note)
        
        # Show matched source if available and score is somewhat relevant
        if c.get("best_source") and score > 0.2:
            table.add_row(
                "", 
                f"[dim]↳ Source: {c['best_source'][:80]}...[/dim]", 
                "", 
                ""
            )
            
    return table


def create_result_panel(title, result, response, source, time_taken=None):
    """Create a panel showing verification results."""
    # Summary stats
    trust_score = result.trust_score
    score_color = "green" if trust_score >= 0.8 else "yellow" if trust_score >= 0.5 else "red"
    verdict = "[green]PASS[/green]" if result.verdict == "PASS" else "[red]FAIL[/red]"
    
    stats_text = (
        f"Verdict: {verdict} | "
        f"Trust Score: [{score_color}]{trust_score:.2f}[/{score_color}] | "
        f"Hallucinations: [bold red]{result.hallucination_count}[/bold red]"
    )
    if time_taken:
        stats_text += f" | Latency: {time_taken:.2f}s"

    group = Group(
        Text("LLM Response:", style="bold cyan"),
        Text(response, style="italic"),
        Text("\nSource Context:", style="bold blue"),
        Text(source, style="dim"),
        Text("\n" + "─" * 40, style="dim"),
        Text(stats_text, justify="center"),
        Text("─" * 40 + "\n", style="dim"),
        format_claim_table(result)
    )
    
    border_color = "green" if result.verdict == "PASS" else "red"
    return Panel(group, title=title, border_style=border_color)


def main():
    console = Console()
    console.clear()
    
    # 1. Header and Config
    console.print(create_header())
    console.print()
    console.print(create_config_panel())
    console.print()
    
    # 2. Init Verifier (Models loading)
    with Live(Panel(Spinner("dots", text="Loading sentence-transformer models..."), border_style="yellow"), console=console, refresh_per_second=10) as live:
        t0 = time.time()
        verifier = CitationVerifier(threshold=0.5)
        # Warm up
        verifier.verify_parallel("test", ["test"])
        t1 = time.time()
        live.update(Panel(f"[green]✅ Models loaded successfully in {t1-t0:.2f}s[/green]", border_style="green"))
    
    console.print()
    time.sleep(3)

    # 3. Scenario 1: Clean Pass
    resp1 = "Water boils at 100°C at standard atmospheric pressure."
    src1 = "Water boils at 100°C at 1 atm pressure."
    
    with Live(Panel(Spinner("dots", text="Verifying Scenario 1..."), title="Scenario 1: Clean Pass", border_style="yellow"), console=console, refresh_per_second=10) as live:
        t0 = time.time()
        res1 = verifier.verify_parallel(resp1, [src1])
        t1 = time.time()
        live.update(create_result_panel("✅ Scenario 1: Clean Pass", res1, resp1, src1, t1-t0))
        
    console.print()
    time.sleep(3)

    # 4. Scenario 2: Obvious Hallucination
    resp2 = "The Eiffel Tower is 330 meters tall and located in Berlin."
    src2 = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall."
    
    with Live(Panel(Spinner("dots", text="Verifying Scenario 2..."), title="Scenario 2: Obvious Hallucination", border_style="yellow"), console=console, refresh_per_second=10) as live:
        t0 = time.time()
        res2 = verifier.verify_parallel(resp2, [src2])
        t1 = time.time()
        live.update(create_result_panel("🚨 Scenario 2: Obvious Hallucination", res2, resp2, src2, t1-t0))

    console.print()
    time.sleep(3)

    # 5. Scenario 3: Subtle Fabrication
    resp3 = "Python was created by James Gosling and released in 1991."
    src3 = "Python was created by Guido van Rossum and first released in 1991."
    
    with Live(Panel(Spinner("dots", text="Verifying Scenario 3..."), title="Scenario 3: Subtle Fabrication", border_style="yellow"), console=console, refresh_per_second=10) as live:
        t0 = time.time()
        res3 = verifier.verify_parallel(resp3, [src3])
        t1 = time.time()
        live.update(create_result_panel("🔍 Scenario 3: Subtle Fabrication", res3, resp3, src3, t1-t0))

    console.print()
    time.sleep(3)
    
    # 6. Batch Summary
    console.print(Panel(
        Align.center(
            "[bold]Batch Verification Summary[/bold]\n\n"
            f"Total Verifications: [cyan]3[/cyan]\n"
            f"Hallucinations Caught: [red]{res1.hallucination_count + res2.hallucination_count + res3.hallucination_count}[/red]\n"
            f"Average Trust: [yellow]{(res1.trust_score + res2.trust_score + res3.trust_score)/3:.2f}[/yellow]\n\n"
            "[dim]Try it: longtracer check \"response\" \"source\"[/dim]"
        ),
        box=box.ROUNDED,
        border_style="blue"
    ))
    console.print()


if __name__ == "__main__":
    main()
