import csv
from datetime import datetime
import os
from pathlib import Path
from pyexpat import model
import yaml
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.console import Console

from dataset import build_dataloaders
from dataset_extras import build_dataloaders_with_extras

# --- DYNAMIC MODEL IMPORTS ---
from models.efficientnet_v2 import EfficientNetV2MStandard, EfficientNetV2MWithExtras
from models.resnet import ResNet50Standard, ResNet50WithExtras
# Add this near your ResNet imports
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights # PyTorch's equivalent to standard MobileNet

from engine import TrainState, train_one_epoch, validate
from utils import seed_everything, ensure_dir, save_checkpoint, load_checkpoint, get_timestamp

# Define accumulation steps (4 * 8 = 32 effective batch size)
ACCUMULATION_STEPS = 1

console = Console()

def create_model(cfg, num_classes, device):
    """Dynamically create either EfficientNetV2-M or ResNet50 based on config."""
    model_type = cfg.get("model", "efficientnet_v2_m").lower()
    use_extras = cfg.get("use_extras", False)
    pretrained = cfg.get("pretrained", True)
    
    # Calculate input channels
    in_chans = 3
    if use_extras:
        in_chans += int(cfg.get("use_fft", True)) + int(cfg.get("use_ela", True)) + int(cfg.get("use_prnu", True))

    console.log(f"[cyan] Building [bold]{model_type.upper()}[/] with {in_chans} input channels")

    # ==========================================
    # 1. EFFICIENTNET V2-M INITIALIZATION
    # ==========================================
    if "efficientnet" in model_type:
        weights_path = Path("weights/efficientnet_v2_m-dc08266a.pth")
        
        def adapt_state_dict(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if "classifier" in k or "head" in k: continue
                if k.startswith("features."): new_state_dict[f"model.{k}"] = v
                else: new_state_dict[k] = v
            return new_state_dict

        if use_extras:
            model = EfficientNetV2MWithExtras(num_classes=num_classes, in_chans=in_chans, pretrained=pretrained)
            if weights_path.exists() and not pretrained:
                state_dict = torch.load(weights_path, map_location='cpu')
                state_dict = adapt_state_dict(state_dict)
                model.load_pretrained_weights(state_dict) # Custom EffNet loader
                console.log(f"[green] Loaded local EfficientNet weights (Adapted keys)[/]")
        else:
            model = EfficientNetV2MStandard(num_classes=num_classes, pretrained=pretrained)
            if weights_path.exists() and not pretrained:
                state_dict = torch.load(weights_path, map_location='cpu')
                state_dict = adapt_state_dict(state_dict)
                model.load_state_dict(state_dict, strict=False)
                console.log(f"[green] Loaded local EfficientNet weights (Adapted keys)[/]")

    # ==========================================
    # 2. RESNET50 INITIALIZATION
    # ==========================================
    elif "resnet50" in model_type:
        
        # --- BASELINE REPLICATION LOGIC ---
        if model_type == "resnet50_baseline":
            console.log(f"[yellow] Initiating Paper 4 Baseline Architecture[/]")
            
            # Load standard pre-trained ResNet50
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            # 1. Freeze the convolutional base weights
            for param in model.parameters():
                param.requires_grad = False
                
            # 2. Replicate the Custom Head from Paper 4
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                # GlobalAveragePooling2D is naturally handled before this layer in PyTorch's ResNet
                nn.Linear(num_ftrs, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, num_classes)
                # Note: Softmax is omitted here because PyTorch's F.cross_entropy includes it
            )
            
            console.log(f"[green] Custom Head Attached: Dense(128)->Dropout(0.5)->Dense(2)[/]")

        # --- YOUR PROPOSED FORENSIC LOGIC ---
        else:
            weights_path = Path("weights/resnet50.pth")
            if use_extras:
                model = ResNet50WithExtras(num_classes=num_classes, in_chans=in_chans, pretrained=pretrained)
                if weights_path.exists() and not pretrained:
                    state_dict = torch.load(weights_path, map_location='cpu')
                    model.load_pretrained_weights(state_dict) 
                    console.log(f"[green] Loaded local ResNet weights safely[/]")
            else:
                model = ResNet50Standard(num_classes=num_classes, pretrained=pretrained)
                if weights_path.exists() and not pretrained:
                    state_dict = torch.load(weights_path, map_location='cpu')
                    model.load_state_dict(state_dict, strict=False)
                    console.log(f"[green] Loaded local ResNet weights safely[/]")

    # ==========================================
    # 3. MOBILENET V2 INITIALIZATION (Paper's Best Model)
    # ==========================================
    elif "mobilenet_v2" in model_type:
        
        if use_extras:
            console.log(f"[yellow] Initiating MobileNetV2 with {in_chans} Forensic Channels[/]")
            model = mobilenet_v2(weights=None) # ImageNet weights don't fit 6 channels
            
            # Rewrite first layer to accept 6 channels
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels=in_chans, out_channels=old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                padding=old_conv.padding, bias=False
            )
        else:
            console.log(f"[yellow] Initiating Paper 4 MobileNetV2 Baseline[/]")
            model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            # Freeze the convolutional base for baseline
            for param in model.parameters():
                param.requires_grad = False
                
        # Replicate Custom Head: Dense(128)->Dropout(0.5)->Dense(2)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(128, num_classes)
        )
        console.log(f"[green] Custom Head Attached to MobileNetV2[/]")

    # ==========================================
    # 4. MOBILENET (V3 Small) INITIALIZATION
    # ==========================================
    elif "mobilenet" in model_type:
        
        if use_extras:
            console.log(f"[yellow] Initiating MobileNet with {in_chans} Forensic Channels[/]")
            model = mobilenet_v3_small(weights=None)
            
            # Rewrite first layer to accept 6 channels
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels=in_chans, out_channels=old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                padding=old_conv.padding, bias=False
            )
        else:
            console.log(f"[yellow] Initiating Paper 4 MobileNet Baseline[/]")
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            # Freeze base for baseline
            for param in model.parameters():
                param.requires_grad = False
                
        # Replicate Custom Head
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
        console.log(f"[green] Custom Head Attached to MobileNet[/]")
        

    # ==========================================
    # 5. MOBILENET V1 INITIALIZATION (via timm)
    # ==========================================
    elif "mobilenet_v1" in model_type:
        import timm 
        
        if use_extras:
            console.log(f"[yellow] Initiating MobileNetV1 with {in_chans} Forensic Channels[/]")
            # timm automatically rewrites the first layer for 6 channels if pretrained=False!
            model = timm.create_model('mobilenetv1_100', pretrained=False, in_chans=in_chans, num_classes=num_classes)
        else:
            console.log(f"[yellow] Initiating Paper MobileNetV1 Baseline[/]")
            # Load with standard ImageNet weights
            model = timm.create_model('mobilenetv1_100', pretrained=True, num_classes=num_classes)
            
            # Freeze the convolutional base to match baseline logic
            for name, param in model.named_parameters():
                # 'classifier' is the name of the final layer in timm's MobileNetV1
                if 'classifier' not in name: 
                    param.requires_grad = False
                    
        console.log(f"[green] MobileNetV1 attached and ready[/]")

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.log(f"[cyan] Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
            
    return model.to(device)

def main():
    parser = argparse.ArgumentParser(description="Train AI Image Detector (Unified)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (e.g., configs/env2.yaml)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f: cfg = yaml.safe_load(f)

    console.log(f"[bold blue] Starting AI Image Detection Training")
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = ensure_dir(Path(cfg.get("output_dir", "outputs")) / "checkpoints")

    # 1. BUILD DATALOADERS
    if cfg.get("use_extras", False):
        train_loader, val_loader, classes = build_dataloaders_with_extras(
            cfg["train_dir"], cfg["val_dir"], cfg["img_size"],
            cfg["batch_size"], cfg["num_workers"],
            cfg.get("rand_augment", True), cfg.get("color_jitter", 0.0), cfg.get("hflip_prob", 0.5),
            cfg.get("use_fft", True), cfg.get("use_ela", True), cfg.get("use_prnu", True)
        )
    else:
        train_loader, val_loader, classes = build_dataloaders(
            cfg["train_dir"], cfg["val_dir"], cfg["img_size"],
            cfg["batch_size"], cfg["num_workers"],
            rand_augment=cfg.get("rand_augment", True), 
            color_jitter=cfg.get("color_jitter", 0.0), 
            hflip_prob=cfg.get("hflip_prob", 0.5)
        )

    # DYNAMICALLY detect all generative model names from the validation set
    val_model_names = sorted(list(set([sample[2] for sample in val_loader.dataset.samples if sample[2] != "Real"])))

    # 2. CREATE OUTPUT DIRECTORY AND CSV
    log_file = Path(cfg.get("output_dir", "outputs")) / "training_log.csv"
    file_exists = log_file.exists()
    
    log_f = open(log_file, "a", newline="")
    csv_writer = csv.writer(log_f)

    if not file_exists:
        header = [
            "Timestamp", "Epoch", "Train_Loss", "Train_Acc", "Train_F1", "Val_Loss", "Val_Acc", 
            "Val_F1", "Val_Precision", "Val_Recall", "Val_FAR", "Val_FRR", "LR"
        ]
        # Append exact model folders found in the dataset
        for name in val_model_names:
            header.append(f"Acc_{name}")
        
        csv_writer.writerow(header)

    # DEBUG: Check how many images were actually loaded
    console.log(f"[yellow] Total Training Images: {len(train_loader.dataset)}")
    console.log(f"[yellow] Total Validation Images: {len(val_loader.dataset)}")
    
    # Count classes in validation
    labels = [sample[1] for sample in val_loader.dataset.samples]
    console.log(f"[cyan] Real images in Val: {labels.count(0)}")
    console.log(f"[cyan] Fake images in Val: {labels.count(1)}")

    # --- PROFESSIONAL CONFIGURATION DASHBOARD ---
    console.rule("[bold green] TRAINING CONFIGURATION")
    console.print(f"[cyan]Model:[/][bold magenta] {cfg.get('model').upper()}")
    console.print(f"[cyan]Device:[/] [bold green]{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}[/]")
    console.print(f"[cyan]Total Epochs:[/] {cfg.get('epochs', 100)}")
    console.print(f"[red]Early Stopping Patience:[/][bold red] {cfg.get('early_stopping_patience', 15)}")
    
    console.print(f"[cyan]Image Pipeline:[/] SmartCropPad -> [bold green]{cfg.get('img_size')}x{cfg.get('img_size')}[/] (No destructive resizing)")
    
    console.print(f"[cyan]Batch Size:[/] {cfg.get('batch_size')} (Accumulation: {ACCUMULATION_STEPS})")
    console.print(f"[cyan]Classes:[/] {len(classes)} {classes}") 
    console.print(f"[blue]Optimizer:[/] {cfg.get('optimizer', 'adamw').upper()}")
    console.print(f"[yellow]Learning Rate:[/] {cfg.get('lr', '1e-4')}")
    console.print(f"[yellow]Forensic Features:[/] FFT: {cfg.get('use_fft')}, ELA: {cfg.get('use_ela')}, PRNU: {cfg.get('use_prnu')}")
    console.print(f"[blue]Generative Models Tracked:[/] {', '.join(val_model_names)}")
    console.rule()
    
    model = create_model(cfg, len(classes), device)
    
    if args.resume: train_state = load_checkpoint(model, ckpt_dir, device)
    else: train_state = None

    total_epochs = cfg.get("epochs", 100)
    lr = float(cfg.get("lr", 1e-3)) # Updated default to match paper's Adam 0.001
    wd = float(cfg.get("weight_decay", 1e-3))
    
    # REPAIR 1: Added RMSprop to match paper's experimental setup
    if cfg["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif cfg["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=float(cfg.get("lr", 1e-4))) # Paper uses 0.0001 for RMSprop
    elif cfg["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        # Fallback
        optimizer = Adam(model.parameters(), lr=lr)
    
    if train_state and train_state.get('optimizer_state'): optimizer.load_state_dict(train_state['optimizer_state'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    if train_state and train_state.get('scheduler_state'): scheduler.load_state_dict(train_state['scheduler_state'])

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get("amp", True) and device.type == "cuda")
    
    start_epoch = 0
    is_fine_tuning = False

    if train_state:
        state = TrainState(epoch=train_state['epoch'], best_val_acc=train_state['best_val_acc'])
        start_epoch = state.epoch
        console.rule("[bold yellow] FINE-TUNING MODE SELECTION")
        if input("Switch to FINE-TUNING (Constant Low LR)? (y/n): ").lower().strip() == 'y':
            is_fine_tuning = True
            for param_group in optimizer.param_groups: param_group['lr'] = float(cfg.get("fine_tune_lr", 1e-5))
            console.print("[bold purple] FINE-TUNING ENABLED!")
    else:
        state = TrainState(epoch=0, best_val_acc=0.0)

    # early stopping
    patience = cfg.get("early_stopping_patience", 10) 
    epochs_no_improve = 0

    # --- MAIN TRAINING LOOP ---
    for epoch in range(start_epoch, total_epochs):
        state.epoch = epoch + 1
        console.log(f"\n[bold cyan]══════ Epoch {state.epoch}/{total_epochs} ══════")

        # 1. TRAIN
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, 
            num_classes=len(classes),
            accumulation_steps=ACCUMULATION_STEPS 
        )

        # 2. VALIDATE
        val_metrics, source_stats = validate(model, val_loader, device, num_classes=len(classes))
        val_loss = val_metrics['loss'] 

        console.log(f" Train → Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        console.log(f" Val   → Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        # --- PER-MODEL FORENSIC REPORT ---
        console.print("\n[bold yellow] PER-MODEL FORENSIC REPORT:")
        for source in val_model_names + ["Real"]:
            if source in source_stats:
                data = source_stats[source]
                acc = data["correct"] / data["total"] if data["total"] > 0 else 0
                color = "green" if acc > 0.85 else "red"
                console.print(f"  {source:15} -> Accuracy: [{color}]{acc:.2%}[/]")
        console.print("") 

        current_lr = optimizer.param_groups[0]['lr']
        
        if not is_fine_tuning: 
            scheduler.step()

        # 3. LOGGING & CHECKPOINT
        log_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), state.epoch,
            f"{train_loss:.4f}", f"{train_metrics['accuracy']:.4f}", f"{train_metrics['f1']:.4f}",
            f"{val_loss:.4f}", f"{val_metrics['accuracy']:.4f}", f"{val_metrics['f1']:.4f}",
            f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}", f"{val_metrics['far']:.4f}", f"{val_metrics['frr']:.4f}",
            f"{current_lr:.2e}"
        ]
        
        # DYNAMICALLY ADD model accuracies
        for name in val_model_names:
            data = source_stats.get(name, {"correct": 0, "total": 1})
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0
            log_row.append(f"{acc:.4f}")

        csv_writer.writerow(log_row)
        log_f.flush()

        #is_best = val_metrics['accuracy'] > state.best_val_acc
        #if is_best:
        #    state.best_val_acc = val_metrics['accuracy']
        #    epochs_no_improve = 0 
        #    console.log(f"[bold green] NEW BEST! Accuracy: {val_metrics['accuracy']:.4f}")
        #else:
        #    epochs_no_improve += 1
        #    console.log(f"[yellow] No improvement for {epochs_no_improve}/{patience} epochs.")

        is_best = val_loss < state.best_val_loss
        if is_best:
            state.best_val_loss = val_loss
            state.best_val_acc = val_metrics['accuracy'] # keep tracking acc for the terminal
            epochs_no_improve = 0 
            console.log(f"[bold green] NEW BEST (Lowest Loss)! Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        save_checkpoint({
            "epoch": state.epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": state.best_val_acc,
            "classes": classes,
            "config": cfg,
            "timestamp": get_timestamp()
        }, is_best=is_best, ckpt_dir=str(ckpt_dir))

        if epochs_no_improve >= patience:
            console.log(f"[bold red] Early stopping triggered! Finished at Epoch {state.epoch}")
            break

    console.log(f"\n[bold green] Training complete! Best Acc: {state.best_val_acc:.4f}")
    log_f.close()

if __name__ == "__main__":
    main()