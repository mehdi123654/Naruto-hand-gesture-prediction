import os
import xmltodict
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Directories
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
IMAGES_DIR = os.path.join(RAW_DIR, "images")
ANNOTATIONS_DIR = os.path.join(RAW_DIR, "annotations")

YOLO_IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
YOLO_LABELS_DIR = os.path.join(PROCESSED_DIR, "labels")

os.makedirs(YOLO_IMAGES_DIR, exist_ok=True)
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Function to convert annotations
def convert_annotation(xml_path, classes):
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())

    img_width = int(doc['annotation']['size']['width'])
    img_height = int(doc['annotation']['size']['height'])

    objects = doc['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]

    annotations = []

    for obj in objects:
        class_name = obj['name']
        if class_name not in classes:
            classes.append(class_name)

        class_id = classes.index(class_name)

        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return annotations

def main():
    classes = []
    data = []

    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]

    for xml_file in xml_files:
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        image_filename = xml_file.replace('.xml', '.jpg')
        annotations = convert_annotation(xml_path, classes)

        data.append((image_filename, annotations))

    df = pd.DataFrame(data, columns=["filename", "annotations"])

    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Create dirs for YOLO format
    for split in ['train', 'val']:
        os.makedirs(f"{YOLO_IMAGES_DIR}/{split}", exist_ok=True)
        os.makedirs(f"{YOLO_LABELS_DIR}/{split}", exist_ok=True)

    # Function to save YOLO format files
    def save_yolo_format(df_split, split_name):
        for _, row in df_split.iterrows():
            src_img_path = os.path.join(IMAGES_DIR, row['filename'])
            dst_img_path = os.path.join(YOLO_IMAGES_DIR, split_name, row['filename'])
            shutil.copy(src_img_path, dst_img_path)

            label_filename = row['filename'].replace('.jpg', '.txt')
            label_path = os.path.join(YOLO_LABELS_DIR, split_name, label_filename)
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(row['annotations']))

    # Save train and validation splits
    save_yolo_format(train_df, 'train')
    save_yolo_format(val_df, 'val')

    # Save class names for YOLO training
    with open(os.path.join(PROCESSED_DIR, "classes.txt"), 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    print("✅ Data prepared successfully!")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    main()
