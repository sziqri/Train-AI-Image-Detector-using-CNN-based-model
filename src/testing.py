import os
import csv
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

from dataset import ForensicMultiSourceDataset as StandardDataset
from dataset_extras import ForensicMultiSourceDataset as ExtrasDataset
from transforms import build_transforms
from transforms_extra import build_transforms_with_extras

# --- DYNAMIC MODEL IMPORTS ---
from models.efficientnet_v2 import EfficientNetV2MStandard, EfficientNetV2MWithExtras
from models.resnet import ResNet50Standard, ResNet50WithExtras
from torchvision.models import mobilenet_v2, mobilenet_v3_small
import torch.nn as nn

console = Console()

def load_checkpoint_and_build_model(ckpt_path: str, device: str):
    """Loads checkpoint, extracts config, and rebuilds the exact architecture."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract config saved during training
    cfg = checkpoint.get('config')
    if not cfg:
        console.log("[red]Error: Checkpoint does not contain config. Ensure you are using the latest train.py format.")
        exit(1)
        
    model_type = cfg.get("model", "resnet").lower()
    use_extras = cfg.get("use_extras", False)
    num_classes = cfg.get("num_classes", 2)
    
    in_chans = 3
    if use_extras:
        in_chans += int(cfg.get("use_fft", True)) + int(cfg.get("use_ela", True)) + int(cfg.get("use_prnu", True))

    console.log(f"[cyan] Rebuilding {model_type.upper()} ({in_chans}-channel) from checkpoint...")

    # 1. EFFICIENTNET
    if "efficientnet" in model_type:
        model = EfficientNetV2MWithExtras(num_classes, in_chans, False) if use_extras else EfficientNetV2MStandard(num_classes, False)
        
    # 2. RESNET
    elif "resnet" in model_type:
        if model_type == "resnet50_baseline":
             from torchvision.models import resnet50
             model = resnet50(weights=None)
             model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))
        else:
            model = ResNet50WithExtras(num_classes, in_chans, False) if use_extras else ResNet50Standard(num_classes, False)

    # 3. MOBILENET V2
    elif "mobilenet_v2" in model_type:
        model = mobilenet_v2(weights=None)
        if use_extras:
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(in_chans, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.classifier[1].in_features, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))

    # 4. MOBILENET
    elif "mobilenet" in model_type:
        model = mobilenet_v3_small(weights=None)
        if use_extras:
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(in_chans, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
        model.classifier[3] = nn.Sequential(nn.Linear(model.classifier[3].in_features, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))

    else:
        raise ValueError(f"Unknown model type in config: {model_type}")

    # Load Weights
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    return model, cfg

def main():
    parser = argparse.ArgumentParser(description="Test AI Image Detector Models")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pth")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Testing batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"[bold green]Initiating Forensic Testing Pipeline on {device}[/]")

    # 1. Load Model & Config
    model, cfg = load_checkpoint_and_build_model(args.ckpt, device)
    
    # 2. Build Dataset & DataLoader
    img_size = cfg.get("img_size", 224)
    use_extras = cfg.get("use_extras", False)
    
    console.log(f"[yellow]Loading test data from {args.test_dir}...[/]")
    
    if use_extras:
        val_transform = build_transforms_with_extras(img_size, train=False)
        test_dataset = ExtrasDataset(
            root_dir=args.test_dir, transform=val_transform,
        )
    else:
        val_transform = build_transforms(img_size, train=False)
        test_dataset = StandardDataset(root_dir=args.test_dir, transform=val_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    classes = ['Real', 'Fake']
    
    # 3. Run Inference
    all_preds = []
    all_labels = []
    all_probs = []

    console.log("[cyan]Running inference across test set...[/]")
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Prob of being Fake (Class 1)

    # 4. Calculate Matrices
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    # Confusion Matrix: tn, fp, fn, tp
    # Assuming 0 = Real (Non-AI), 1 = Fake (AI)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    overall_acc = (tp + tn) / (tp + tn + fp + fn)
    
    # Matrices for Fake (AI) Class
    precision_ai = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_ai = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_ai = 2 * (precision_ai * recall_ai) / (precision_ai + recall_ai) if (precision_ai + recall_ai) > 0 else 0
    
    # Matrices for Real (Non-AI) Class
    precision_non_ai = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_non_ai = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_non_ai = 2 * (precision_non_ai * recall_non_ai) / (precision_non_ai + recall_non_ai) if (precision_non_ai + recall_non_ai) > 0 else 0

    # Advanced Forensic Metrics
    far = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Acceptance Rate (Real classified as AI)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0 # False Rejection Rate (AI classified as Real)

    # 5. Display Console Report
    console.print("\n[bold yellow]=====================================================[/]")
    console.print(f"[bold yellow]  FORENSIC TESTING REPORT: {cfg.get('model').upper()}[/]")
    console.print("[bold yellow]=====================================================[/]\n")
    
    console.print(f"[bold white]Overall Accuracy:[/] {overall_acc*100:.2f}%")
    console.print(f"[bold white]False Acceptance Rate (FAR):[/] {far*100:.2f}%")
    console.print(f"[bold white]False Rejection Rate (FRR):[/] {frr*100:.2f}%\n")

    table = Table(title="Class-Specific Matrices")
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Precision", justify="right", style="magenta")
    table.add_column("Recall", justify="right", style="green")
    table.add_column("F1-Score", justify="right", style="blue")
    
    table.add_row("Real (Non-AI)", f"{precision_non_ai:.4f}", f"{recall_non_ai:.4f}", f"{f1_non_ai:.4f}")
    table.add_row("Fake (AI)", f"{precision_ai:.4f}", f"{recall_ai:.4f}", f"{f1_ai:.4f}")
    console.print(table)
    
    # 6. Save to CSV
    # Create logs directory if it doesn't exist
    Path("test_logs").mkdir(exist_ok=True)
    
    model_name = Path(args.ckpt).parent.parent.name # e.g. outputs/resnet50_forensic/checkpoints/best.pth -> resnet50_forensic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"test_logs/test_matrix_{model_name}_{timestamp}.csv"
    
    headers = [
        "Model", "Timestamp", "Total_Images", "Overall_Accuracy", 
        "Precision_Real", "Recall_Real", "F1_Real",
        "Precision_AI", "Recall_AI", "F1_AI",
        "FAR", "FRR", "TP", "TN", "FP", "FN"
    ]
    
    data = [
        model_name, timestamp, len(y_true), f"{overall_acc:.4f}",
        f"{precision_non_ai:.4f}", f"{recall_non_ai:.4f}", f"{f1_non_ai:.4f}",
        f"{precision_ai:.4f}", f"{recall_ai:.4f}", f"{f1_ai:.4f}",
        f"{far:.4f}", f"{frr:.4f}", tp, tn, fp, fn
    ]
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(data)
        
    console.print(f"\n[bold green]✓ Testing Complete. Detailed matrix saved to: {csv_path}[/]")

if __name__ == "__main__":
    main()