from ultralytics import YOLO
import os

def main():
    # Define directories
    PROCESSED_DIR = "../data/processed"

    # Load class names
    with open(os.path.join(PROCESSED_DIR, "classes.txt")) as f:
        class_names = [line.strip() for line in f.readlines()]

    # Create data.yaml with hardcoded clean paths
    data_yaml_content = f"""
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""

    yaml_path = os.path.join(PROCESSED_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(data_yaml_content)

    print("✅ data.yaml created successfully!")

    # Initialize model
    model = YOLO("yolov8m.pt")  # instead of yolov8s.pt


    # Start training
    model.train(
    data=os.path.abspath(yaml_path),
    epochs=50,             # Maximum if needed
    imgsz=640,
    batch=16,
    project="../models",
    name="naruto_jutsu_detector_beast",
    optimizer="AdamW",       # Better generalization
    patience=20,             # ⏳ Early stopping patience (stop if no val improvement after 20 epochs)
    lr0=0.003,               # Lower learning rate to stabilize
    close_mosaic=5,          # Stop heavy augmentations after 5 epochs
    mosaic=1.0,              # Mosaic enabled early
    mixup=0.2,               # Mixup augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,              # Rotation augmentation
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    fliplr=0.5,              # Flip horizontally
    save_period=-1,          # Save only best checkpoints
    save=True,
    val=True,                # Validate after each epoch
    verbose=True,            # See nice output
    deterministic=True       # Reproducibility
    )



    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
