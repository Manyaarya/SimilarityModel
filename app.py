import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

model = YOLO('yolov8n.pt')

# Function to extract features using a pre-trained ResNet model
def extract_features(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resnet_model = models.resnet50(pretrained=True)
    resnet_model.eval()

    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = resnet_model(image)

    return features.squeeze().numpy()

@app.post("/find_similar/")
async def find_similar(file: UploadFile = File(...)):
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Load and process the image
    results = model(source=image_path, conf=0.8)

    # Directory to save cropped images
    cropped_images_dir = "cropped_images"
    os.makedirs(cropped_images_dir, exist_ok=True)

    # Crop detected objects from the image
    for i, result in enumerate(results):
        img = result.orig_img
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            cropped_img_path = os.path.join(cropped_images_dir, f"cropped_{i}_{j}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)

    # Extract features for all cropped images
    feature_dict = {}
    for img_path in os.listdir(cropped_images_dir):
        full_img_path = os.path.join(cropped_images_dir, img_path)
        if os.path.isfile(full_img_path):
            features = extract_features(full_img_path)
            feature_dict[img_path] = features

    # Load catalog images and extract features
    catalog_dir = "ikea-master/images"  # Adjust this path
    catalog_features = {}

    for category in os.listdir(catalog_dir):
        category_path = os.path.join(catalog_dir, category)
        if os.path.isdir(category_path):
            for img_path in os.listdir(category_path):
                full_img_path = os.path.join(category_path, img_path)
                if os.path.isfile(full_img_path):
                    features = extract_features(full_img_path)
                    catalog_features[f"{category}/{img_path}"] = features

    # Find similar items
    def find_similar_items(features, catalog_features, top_k=5):
        similarities = []
        for img_path, catalog_feature in catalog_features.items():
            similarity = cosine_similarity(features.reshape(1, -1), catalog_feature.reshape(1, -1))[0][0]
            similarities.append((img_path, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    recommendations = {}
    for img_name, features in feature_dict.items():
        similar_items = find_similar_items(features, catalog_features)
        recommendations[img_name] = similar_items

    return JSONResponse(content=recommendations)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)