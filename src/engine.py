# src/engine.py
from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, 
    MulticlassPrecision, MulticlassRecall,
    MulticlassConfusionMatrix
)
from rich.console import Console

console = Console()

@dataclass
class TrainState:
    epoch: int = 0
    best_val_acc: float = 0.0
    best_f1: float = 0.0
    best_val_loss: float = float('inf') # Add this line

class ForensicMetrics:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        
        self.accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        
    def update(self, logits, targets):
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, targets)
        self.f1.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.confusion_matrix.update(preds, targets)
    
    def compute(self) -> Dict[str, float]:
        accuracy = self.accuracy.compute().item()
        f1 = self.f1.compute().item()
        precision = self.precision.compute().item()
        recall = self.recall.compute().item()
        
        cm = self.confusion_matrix.compute()
        far, frr = self._compute_far_frr(cm)
        
        return {
            'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall,
            'far': far, 'frr': frr
        }
    
    def _compute_far_frr(self, confusion_matrix):
        if self.num_classes == 2:
            # Standard: 0 = Real (Negative), 1 = Fake (Positive)
            tn = confusion_matrix[0, 0].item() # Real correctly identified
            fp = confusion_matrix[0, 1].item() # Real mistaken for Fake (False Alarm)
            fn = confusion_matrix[1, 0].item() # Fake mistaken for Real (Missed Detection)
            tp = confusion_matrix[1, 1].item() # Fake correctly identified
            
            # FAR: Probability that a FAKE image is missed (Forensic definition)
            far = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            # FRR: Probability that a REAL image is wrongly accused
            frr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            return float(far), float(frr)

# FIXED FUNCTION BELOW
def train_one_epoch(model, loader, optimizer, scaler, device, num_classes, accumulation_steps=1):
    model.train()
    metrics = ForensicMetrics(num_classes, device)
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, (x, y, sources) in enumerate(loader): # Unpack 3 values
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(x)
            loss = F.cross_entropy(logits, y) / accumulation_steps
        if scaler: scaler.scale(loss).backward()
        else: loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            if scaler: scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * accumulation_steps
        metrics.update(logits, y)
    return total_loss / len(loader), metrics.compute()

def validate(model, loader, device, num_classes):
    model.eval()
    metrics = ForensicMetrics(num_classes, device)
    total_loss, source_stats = 0.0, {}
    with torch.no_grad():
        for x, y, sources in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y).item()
            metrics.update(logits, y)
            preds = torch.argmax(logits, dim=1)
            for p, t, s in zip(preds, y, sources):
                if s not in source_stats: source_stats[s] = {"correct": 0, "total": 0}
                if p == t: source_stats[s]["correct"] += 1
                source_stats[s]["total"] += 1
    val_metrics = metrics.compute()
    val_metrics['loss'] = total_loss / len(loader)
    return val_metrics, source_stats