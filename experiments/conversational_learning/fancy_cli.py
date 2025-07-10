"""
Fancy CLI Interface for Neural Plasticity Learning

Creates a multi-pane terminal interface:
- Bottom: Stats pane (network status, learning metrics)
- Top-right: Conversation pane (clean conversation flow)
- Top-left: Monitoring pane (logs, visualizations, insights)
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import math

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich import box
from rich.markdown import Markdown
from rich.syntax import Syntax


class FancyCLI:
    """
    Multi-pane CLI interface for neural learning visualization.
    
    Layout:
    ┌─────────────────────────────────────────────────────┐
    │ Monitoring Pane          │ Conversation Pane        │
    │ • Logs                   │ • Teacher messages       │
    │ • Network insights       │ • Learner responses      │
    │ • Learning visualizations│ • Clean conversation     │
    ├─────────────────────────────────────────────────────┤
    │ Stats Pane (Network status, metrics, progress)     │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(self, max_conversation_lines: int = 20, max_log_lines: int = 15):
        self.console = Console()
        self.max_conversation_lines = max_conversation_lines
        self.max_log_lines = max_log_lines
        
        # Data storage
        self.conversation_history = deque(maxlen=max_conversation_lines)
        self.log_entries = deque(maxlen=max_log_lines)
        self.current_stats = {}
        self.learning_timeline = deque(maxlen=100)  # For visualizations
        
        # Layout setup
        self.layout = Layout()
        self._setup_layout()
        
        # State tracking
        self.is_running = False
        self.last_update = time.time()
        
    def _setup_layout(self):
        """Setup the multi-pane layout."""
        # Create main layout
        self.layout.split_column(
            Layout(name="main", ratio=3),
            Layout(name="stats", size=8)
        )
        
        # Split main area horizontally
        self.layout["main"].split_row(
            Layout(name="monitoring", ratio=1),
            Layout(name="conversation", ratio=1)
        )
    
    def start(self):
        """Start the fancy CLI interface."""
        self.is_running = True
        self.live = Live(self.layout, console=self.console, refresh_per_second=2)
        self.live.start()
        
        # Initial welcome
        self.log("🎯 Neural Plasticity Learning Interface Started", "system")
        self.log("📊 Multi-pane visualization active", "system")
    
    def stop(self):
        """Stop the CLI interface."""
        self.is_running = False
        if hasattr(self, 'live'):
            self.live.stop()
    
    def update_display(self):
        """Update all panes with current data."""
        if not self.is_running:
            return
            
        try:
            # Update each pane with error handling
            self.layout["monitoring"].update(self._create_monitoring_pane())
            self.layout["conversation"].update(self._create_conversation_pane())
            self.layout["stats"].update(self._create_stats_pane())
            
            self.last_update = time.time()
        except Exception as e:
            # Fallback to simple text display if rendering fails
            error_panel = Panel(
                f"[red]Display error: {str(e)}[/red]\n[dim]Continuing with basic display...[/dim]",
                title="⚠️ Display Issue", 
                border_style="red"
            )
            self.layout["monitoring"].update(error_panel)
    
    def _create_monitoring_pane(self) -> Panel:
        """Create the monitoring pane with logs and visualizations."""
        content = []
        
        # Recent logs section
        if self.log_entries:
            content.append("📋 [bold blue]Recent Activity[/bold blue]")
            content.append("")
            
            for entry in list(self.log_entries)[-10:]:
                timestamp, message, category = entry
                time_str = timestamp.strftime("%H:%M:%S")
                
                # Color code by category
                color = {
                    "system": "cyan",
                    "learning": "green", 
                    "warning": "yellow",
                    "error": "red",
                    "plasticity": "magenta",
                    "insight": "blue"
                }.get(category, "white")
                
                content.append(f"[dim]{time_str}[/dim] [{color}]{message}[/{color}]")
            
            content.append("")
        
        # Learning insights section
        if self.current_stats:
            content.append("🧠 [bold green]Learning Insights[/bold green]")
            content.append("")
            
            # Network maturity indicator
            maturity = self.current_stats.get('maturity_factor', 1.0)
            if maturity > 0.8:
                content.append("🌱 [yellow]Early learning phase - high plasticity[/yellow]")
            elif maturity > 0.3:
                content.append("🌿 [green]Active learning phase - moderate plasticity[/green]")
            else:
                content.append("🌳 [blue]Mature phase - stable connections[/blue]")
            
            # Learning rate status
            ltp_rate = self.current_stats.get('current_ltp_rate', 0)
            if ltp_rate > 0.008:
                content.append("⚡ [bold green]High learning rate - rapid adaptation[/bold green]")
            elif ltp_rate > 0.005:
                content.append("🔄 [green]Moderate learning rate - steady progress[/green]")
            else:
                content.append("🐌 [yellow]Low learning rate - fine-tuning mode[/yellow]")
            
            # Association quality
            overlap = self.current_stats.get('response_feedback_overlap', 0)
            if overlap > 0.3:
                content.append("🎯 [bold green]Strong context-response associations[/bold green]")
            elif overlap > 0.15:
                content.append("🎪 [green]Developing associations[/green]")
            else:
                content.append("🎲 [yellow]Exploring - weak associations[/yellow]")
        
        # Simple connectivity visualization
        if self.learning_timeline:
            content.append("")
            content.append("📊 [bold cyan]Connectivity Trend[/bold cyan]")
            try:
                content.append(self._create_mini_chart())
            except Exception:
                content.append("[dim]Chart temporarily unavailable[/dim]")
        
        text_content = "\n".join(content) if content else "[dim]Monitoring data loading...[/dim]"
        
        return Panel(
            text_content,
            title="🔍 Neural Monitoring",
            border_style="cyan",
            padding=(0, 1)
        )
    
    def _create_conversation_pane(self) -> Panel:
        """Create the conversation pane with clean dialogue."""
        content = []
        
        if self.conversation_history:
            for entry in self.conversation_history:
                role, message, timestamp = entry
                time_str = timestamp.strftime("%H:%M:%S")
                
                if role == "teacher":
                    content.append(f"[dim]{time_str}[/dim] 👩‍🏫 [bold blue]Teacher:[/bold blue] {message}")
                elif role == "learner":
                    content.append(f"[dim]{time_str}[/dim] 🤖 [bold green]Network:[/bold green] {message}")
                elif role == "system":
                    content.append(f"[dim]{time_str}[/dim] ⚙️  [dim]{message}[/dim]")
                
                content.append("")
        else:
            content.append("[dim]Conversation starting...[/dim]")
        
        text_content = "\n".join(content)
        
        return Panel(
            text_content,
            title="💬 Learning Conversation",
            border_style="green",
            padding=(0, 1)
        )
    
    def _create_stats_pane(self) -> Panel:
        """Create the stats pane with network metrics."""
        if not self.current_stats:
            return Panel(
                "[dim]Statistics loading...[/dim]",
                title="📊 Network Statistics", 
                border_style="blue"
            )
        
        # Create metrics table
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=15)
        table.add_column("Metric", style="cyan", width=20) 
        table.add_column("Value", style="white", width=15)
        
        stats = self.current_stats
        
        # Format metrics nicely
        metrics = [
            ("🧠 Neurons", f"{stats.get('network_neurons', 0):,}"),
            ("🔗 Connections", f"{stats.get('connectivity', 0):.1%}"),
            ("⚡ Active", f"{stats.get('active_neurons', 0):,}"),
            ("📚 Vocabulary", f"{stats.get('vocabulary_size', 0):,}"),
            ("🔄 Learning Step", f"{stats.get('plasticity_step', 0):,}"),
            ("💬 Interactions", f"{stats.get('total_interactions', 0):,}"),
            ("🎯 Associations", f"{stats.get('learned_associations', 0):,}"),
            ("📈 LTP Rate", f"{stats.get('current_ltp_rate', 0):.4f}"),
        ]
        
        # Add rows in pairs
        for i in range(0, len(metrics), 2):
            row = [metrics[i][0], metrics[i][1]]
            if i + 1 < len(metrics):
                row.extend([metrics[i + 1][0], metrics[i + 1][1]])
            else:
                row.extend(["", ""])
            table.add_row(*row)
        
        # Learning progress indicator
        progress_text = ""
        interactions = stats.get('total_interactions', 0)
        if interactions < 100:
            progress_text = "🌱 [yellow]Bootstrapping[/yellow]"
        elif interactions < 500:
            progress_text = "🌿 [green]Early Learning[/green]"
        elif interactions < 2000:
            progress_text = "🌳 [blue]Active Learning[/blue]"
        else:
            progress_text = "🎓 [magenta]Advanced Learning[/magenta]"
        
        # Combine table and progress
        content = Columns([table, Align.center(progress_text)])
        
        return Panel(
            content,
            title="📊 Neural Network Statistics",
            border_style="blue",
            padding=(0, 1)
        )
    
    def _create_mini_chart(self) -> str:
        """Create a simple ASCII chart of connectivity over time."""
        if len(self.learning_timeline) < 2:
            return "[dim]Collecting data...[/dim]"
        
        # Get recent connectivity values
        values = [point['connectivity'] for point in list(self.learning_timeline)[-20:]]
        if not values:
            return "[dim]No data[/dim]"
        
        # Normalize to 0-7 range for display (bars has 8 characters, index 0-7)
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            normalized = [4] * len(values)  # Middle value when all same
        else:
            # Ensure values are within bounds [0, 7]
            normalized = [min(7, max(0, int(7 * (v - min_val) / (max_val - min_val)))) for v in values]
        
        # Create bar chart using Unicode blocks
        bars = "▁▂▃▄▅▆▇█"
        chart = "".join(bars[n] for n in normalized)
        
        return f"{chart} [dim]({min_val:.2%} - {max_val:.2%})[/dim]"
    
    # Public interface methods
    def add_conversation(self, role: str, message: str):
        """Add a conversation entry."""
        self.conversation_history.append((role, message, datetime.now()))
        if self.is_running:
            self.update_display()
    
    def log(self, message: str, category: str = "system"):
        """Add a log entry."""
        self.log_entries.append((datetime.now(), message, category))
        if self.is_running:
            self.update_display()
    
    def update_stats(self, stats: Dict):
        """Update the statistics."""
        self.current_stats = stats.copy()
        
        # Add to timeline for visualizations
        self.learning_timeline.append({
            'timestamp': time.time(),
            'connectivity': stats.get('connectivity', 0),
            'active_neurons': stats.get('active_neurons', 0),
            'interactions': stats.get('total_interactions', 0)
        })
        
        if self.is_running:
            self.update_display()
    
    def log_plasticity_event(self, event_type: str, details: str):
        """Log a plasticity-related event."""
        self.log(f"{event_type}: {details}", "plasticity")
    
    def log_learning_insight(self, insight: str):
        """Log a learning insight."""
        self.log(insight, "insight")
    
    def log_warning(self, warning: str):
        """Log a warning."""
        self.log(warning, "warning")
    
    def log_error(self, error: str):
        """Log an error."""
        self.log(error, "error")


# Convenience context manager
class CLIManager:
    """Context manager for the fancy CLI."""
    
    def __init__(self, **kwargs):
        self.cli = FancyCLI(**kwargs)
    
    def __enter__(self):
        self.cli.start()
        return self.cli
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cli.stop()