from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Skin Condition Analyzer API")

# Load model once at startup, not per-request
MODEL_PATH = "best_model.keras"
model = load_model(MODEL_PATH)

CLASS_NAMES = ["acne", "blackheades", "dark spots", "pores", "wrinkles"]
IMG_SIZE = (224, 224)

@app.get("/")
def root():
    return {
        "message": "Skin Condition Analyzer API",
        "endpoints": {
            "health_check": "/health",
            "predict": "/predict (POST, multipart/form-data, field name: file)",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process the uploaded file as an image")

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_idx])

    return JSONResponse({
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 4),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }
    })