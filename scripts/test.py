import cv2
import os

folder = "../data/raw/images"
sizes = set()

for image_name in os.listdir(folder):
    if image_name.endswith('.jpg'):
        img = cv2.imread(os.path.join(folder, image_name))
        sizes.add(img.shape[:2])  # (height, width)

print("Unique image sizes:", sizes)
