import os
import torch
import random
import numpy as np
from pathlib import Path
from rich.console import Console
from typing import Union
from datetime import datetime

console = Console()

def get_timestamp():
    """Get current timestamp for versioning."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def seed_everything(seed: int = 42):
    """For reproducibility - essential for research!"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_checkpoint(model, ckpt_dir, device="cuda"):
    """
    Interactively ask the user whether to load 'last.pth' or 'best.pth'
    if both exist.
    """
    ckpt_dir = Path(ckpt_dir)
    last_path = ckpt_dir / "last.pth"
    best_path = ckpt_dir / "best.pth"
    target_path = None

    # Case 1: Both exist -> ASK USER
    if last_path.exists() and best_path.exists():
        console.rule("[bold yellow] RESUME CHECKPOINT SELECTION")
        console.print(f"[cyan]Found two checkpoints in {ckpt_dir}:")
        
        # Load meta-data to show epoch numbers (Optional but helpful)
        try:
            last_meta = torch.load(last_path, map_location="cpu")
            best_meta = torch.load(best_path, map_location="cpu")
            last_epoch = last_meta.get('epoch', '?')
            best_epoch = best_meta.get('epoch', '?')
            best_acc = best_meta.get('best_val_acc', 0.0)
        except:
            last_epoch, best_epoch, best_acc = "?", "?", "?"

        console.print(f"1. [bold green]LAST[/] (last.pth) -> Epoch {last_epoch} (Resume training here!)")
        console.print(f"2. [bold gold1]BEST[/] (best.pth) -> Epoch {best_epoch} (Acc: {best_acc:.4f})")
        console.print("")

        while True:
            choice = input("Type 'last' or 'best' to select: ").strip().lower()
            if choice in ['last', 'l']:
                target_path = last_path
                break
            elif choice in ['best', 'b']:
                target_path = best_path
                break
            else:
                console.print("[red]Invalid choice. Please type 'last' or 'best'.")

    # Case 2: Only Last exists
    elif last_path.exists():
        target_path = last_path
    
    # Case 3: Only Best exists
    elif best_path.exists():
        target_path = best_path
        
    # Case 4: None exist
    else:
        console.print("[yellow] No checkpoint found. Starting fresh.")
        return None

    # --- Perform the Load ---
    console.print(f"[bold green] Loading checkpoint: {target_path.name}")
    checkpoint = torch.load(target_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    
    # Return state for optimizer/scheduler loading
    train_state = {
        'epoch': checkpoint.get("epoch", 0),
        'best_val_acc': checkpoint.get("best_val_acc", 0.0),
        'optimizer_state': checkpoint.get("optimizer_state"),
        'scheduler_state': checkpoint.get("scheduler_state")
    }
    
    console.print(f"[green] Continuing from epoch {train_state['epoch']}")
    return train_state

def save_checkpoint(state, is_best: bool, ckpt_dir: str, filename: str = "last.pth"):
    """Save checkpoint with versioning for best models."""
    ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    
    if is_best:
        best_path = os.path.join(ckpt_dir, "best.pth")
        torch.save(state, best_path)
        
        # Save versioned copy for forensic trail
        accuracy = state.get("best_val_acc", 0)
        timestamp = state.get("timestamp", get_timestamp())  # FIXED: Use get_timestamp() if not provided
        version_path = os.path.join(ckpt_dir, f"best_acc_{accuracy:.4f}_{timestamp}.pth")
        torch.save(state, version_path)
        
        console.log(f"[green] New best model! → {best_path}")
        console.log(f"[green] Version saved → {version_path}")
    else:
        console.log(f"[yellow] Saved checkpoint → {path}")