from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import io
import pickle
import uvicorn
import os
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with your Flutter frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Models ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load segmentation model (.h5)
seg_model_path = os.path.join(BASE_DIR, "palm_segmentation_model.h5")
seg_model = load_model(seg_model_path, compile=False)

# Load classifier (.pkl)
clf_model_path = os.path.join(BASE_DIR, "RF-balancedPalm.pkl")
with open(clf_model_path, "rb") as f:
    model = pickle.load(f)

# === Helper Functions ===

def preprocess_for_segmentation(image_data, target_size=(128, 128)):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize(target_size)
    image_np = np.array(image) / 255.0
    return image_np

def segment_palm(image_np):
    input_img = np.expand_dims(image_np, axis=0)  # (1, 128, 128, 3)
    prediction = seg_model.predict(input_img)[0]

    mask = (prediction > 0.5).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

    segmented = (image_np * mask[:, :, np.newaxis]).astype(np.float32)
    return segmented

def flatten_image(image_np):
    resized = cv2.resize(image_np, (224, 224))  # Resize to match model input
    normalized = resized / 1.0
    flat = normalized.flatten().reshape(1, -1)
    return flat

# === API Endpoints ===

@app.post("/predict")
async def predict_anemia(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "Invalid file type. Please upload an image."}

        contents = await file.read()

        # Step 1: Segment palm
        img_np = preprocess_for_segmentation(contents)
        segmented = segment_palm(img_np)

        # Check if segmentation mask failed
        if np.sum(segmented) == 0:
            return {"error": "Palm region could not be detected. Try a clearer image."}

        # Step 2: Flatten for classification
        flat_input = flatten_image(segmented)

        # Step 3: Predict
        prediction = model.predict(flat_input)
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

# === Run Server ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
