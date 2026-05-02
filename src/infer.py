import argparse
from pathlib import Path
import json
import torch
from PIL import Image

# --- DYNAMIC MODEL IMPORTS ---
from models.efficientnet_v2 import EfficientNetV2MStandard, EfficientNetV2MWithExtras
from models.resnet import ResNet50Standard, ResNet50WithExtras

from transforms import build_transforms
from transforms_extra import build_transforms_with_extras

def load_checkpoint_and_config(ckpt_path: str, device: str):
    """Loads the model and automatically extracts training configurations."""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 1. Auto-extract configuration saved during training
    config = ckpt.get("config", {})
    classes = ckpt.get("classes", ["Real", "AI-Generated"])  # Fallback to binary
    num_classes = len(classes)
    
    use_extras = config.get("use_extras", False)
    use_fft = config.get("use_fft", False)
    use_ela = config.get("use_ela", False)
    use_prnu = config.get("use_prnu", False)
    
    # 2. Dynamically calculate input channels
    in_chans = 3
    if use_extras:
        in_chans += int(use_fft) + int(use_ela) + int(use_prnu)

    # 3. Dynamically build the correct architecture
    model_type = config.get("model", "efficientnet_v2_m").lower()
    
    if "efficientnet" in model_type:
        if use_extras:
            model = EfficientNetV2MWithExtras(num_classes=num_classes, in_chans=in_chans, pretrained=False)
        else:
            model = EfficientNetV2MStandard(num_classes=num_classes, pretrained=False)
    elif "resnet" in model_type:
        if use_extras:
            model = ResNet50WithExtras(num_classes=num_classes, in_chans=in_chans, pretrained=False)
        else:
            model = ResNet50Standard(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model type in checkpoint config: {model_type}")
    
    # 4. Load weights
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    
    # Debug info for the terminal/GUI console
    print(f" Loaded Checkpoint: {Path(ckpt_path).name}")
    print(f"   ├─ Model Architecture: {model_type.upper()}")
    print(f"   ├─ Classes: {num_classes} {classes}")
    print(f"   ├─ Channels: {in_chans}")
    print(f"   └─ Forensic Features -> FFT: {use_fft} | ELA: {use_ela} | PRNU: {use_prnu}")
    
    return model, classes, config

def main():
    parser = argparse.ArgumentParser(description="AI Image Detection Inference for GUI")
    parser.add_argument("--paths", type=str, nargs="+", required=True, help="Image paths to analyze")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--topk", type=int, default=2, help="Number of top predictions to show")
    
    args = parser.parse_args()
    device = torch.device(args.device)

    # Load model and auto-detect settings
    model, classes, config = load_checkpoint_and_config(args.ckpt, device="cpu")
    model.to(device)

    # Read the exact image size used during training (Defaults to 384 if not found)
    img_size = config.get("img_size", 384)

    # Select the appropriate transform using the extracted config
    if config.get("use_extras", False):
        tfm = build_transforms_with_extras(
            img_size=img_size, 
            train=False,
            use_fft=config.get("use_fft", False), 
            use_ela=config.get("use_ela", False), 
            use_prnu=config.get("use_prnu", False)
        )
    else:
        tfm = build_transforms(img_size=img_size, train=False)

    # Load and process images
    paths = [Path(p) for p in args.paths]
    results = {}
    
    print(f"\n Analyzing {len(paths)} image(s) via SmartCropPad ({img_size}x{img_size})...")
    
    with torch.no_grad():
        for img_path in paths:
            try:
                img = Image.open(img_path).convert("RGB")
                
                # Apply the non-destructive transform pipeline
                x = tfm(img).unsqueeze(0).to(device)
                
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                conf, idx = torch.topk(probs, k=min(args.topk, len(classes)), dim=-1)
                
                conf, idx = conf.squeeze(0).cpu().tolist(), idx.squeeze(0).cpu().tolist()
                labels = [classes[i] for i in idx]
                
                # Format for GUI JSON parsing
                results[str(img_path)] = {
                    "status": "success",
                    "predictions": [{"label": l, "confidence": float(c)} for l, c in zip(labels, conf)],
                    "pipeline": f"CenterCrop/Pad {img_size}x{img_size}"
                }
                
                # Print immediate results for terminal feedback
                print(f" {img_path.name}: {labels[0]} ({conf[0]:.2%})")
                
            except Exception as e:
                print(f" Error processing {img_path}: {e}")
                results[str(img_path)] = {"status": "error", "error_message": str(e)}

    # Output final results in strictly formatted JSON for the GUI backend to parse
    print("\n" + "="*50)
    print("FINAL_JSON_OUTPUT_START")
    print(json.dumps(results, indent=2))
    print("FINAL_JSON_OUTPUT_END")
    print("="*50)

if __name__ == "__main__":
    main()