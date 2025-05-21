from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import cv2
from PIL import Image
import io
import os
import joblib  # use joblib for loading sklearn models
from tensorflow.keras.models import load_model
import uvicorn

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
seg_model_path = os.path.join(BASE_DIR, "palm_segmentation_model.h5")
clf_model_path = os.path.join(BASE_DIR, "RF-balancedPalm.pkl")

# Load models
seg_model = load_model(seg_model_path, compile=False)
clf_model = joblib.load(clf_model_path)

# ========= Helper Functions =========

def preprocess_for_segmentation(image_data, target_size=(128, 128)):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize(target_size)
    image_np = np.array(image) / 255.0
    return image_np

def segment_palm(image_np):
    input_img = np.expand_dims(image_np, axis=0)  # shape: (1, 128, 128, 3)
    prediction = seg_model.predict(input_img)[0]

    mask = (prediction > 0.5).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    segmented = (image_np * mask[:, :, np.newaxis]).astype(np.float32)
    return segmented

def extract_color_histogram(image_np, bins=64):
    # Expecting image_np in range [0,1], shape: (128,128,3)
    image_uint8 = (image_np * 255).astype(np.uint8)

    hist_features = []
    for channel in range(3):  # RGB channels
        hist = cv2.calcHist([image_uint8], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)

    # Total features: 64 * 3 = 192
    # Pad or repeat to reach 512
    if len(hist_features) < 512:
        repeats = (512 + len(hist_features) - 1) // len(hist_features)
        hist_features = (hist_features * repeats)[:512]

    return np.array(hist_features).reshape(1, -1)

# ========= API Endpoints =========

@app.post("/predict")
async def predict_anemia(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "Invalid file type. Please upload an image."}



        contents = await file.read()

        # Palm Segmentation
        img_np = preprocess_for_segmentation(contents)
        segmented = segment_palm(img_np)

        if np.sum(segmented) == 0:
            return {"error": "Palm region could not be detected. Try a clearer image."}

        # Feature Extraction
        features = extract_color_histogram(segmented)

        # Prediction
        prediction = clf_model.predict(features)
        label = "Anemic" if prediction[0] == 1 else "Non-Anemic"

        return {"label": label}

    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}

@app.get("/")
async def serve_html():
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

# ========= Run Server =========

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
