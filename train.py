from ultralytics import YOLO
import torch
import multiprocessing

def main():
    # Auto-select GPU if available
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Load model
    model = YOLO("yolo11n.pt")

    # Train
    model.train(
        data="data/data.yaml",        # your dataset config
        epochs=100,                   # full training schedule
        batch=16,                     # fits on your 10 GB RTX 3080
        imgsz=640,                    # image size
        device=device,                # auto-GPU or CPU
        workers=0,                    # on Windows, set to 0 to avoid spawn issues
        cache="disk",                 # persistent disk cache for faster reloads
        project="runs/train",         # where to save runs
        name="yolo11n_bottles_v1",    # your run name/version
        exist_ok=True,                # overwrite if re-running same name
    )

if __name__ == "__main__":
    # Necessary for safe multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
