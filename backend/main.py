from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from annotation import draw_annotations, encode_image
from api_models import ConstellationResponse, IdentifyResponse
from vision.constellation_match import ConstellationMatcher
from vision.catalog import load_star_catalog
from vision.preprocess import preprocess_image
from vision.star_detection import detect_stars

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "data" / "star_catalog.json"

matcher = ConstellationMatcher(load_star_catalog(CATALOG_PATH))

app = FastAPI(title="AI Constellation & Object Identifier", version="1.0.0")

local_dev_origin_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=local_dev_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - Pillow exceptions vary by platform.
        raise HTTPException(status_code=400, detail="Could not read image.") from exc

    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed = preprocess_image(image_array)
    detection = detect_stars(processed)
    matches = matcher.match(detection.stars, image_array.shape[1], image_array.shape[0])
    annotated = draw_annotations(image_array, detection, matches)
    annotated_b64 = encode_image(annotated)

    response = IdentifyResponse(
        constellations=[
            ConstellationResponse(name=match.name, confidence=round(match.confidence, 3))
            for match in matches
        ],
        stars_detected=len(detection.stars),
        possible_planets=[
            {
                "x": round(planet["x"], 2),
                "y": round(planet["y"], 2),
                "brightness": round(planet["brightness"], 2),
            }
            for planet in detection.possible_planets
        ],
        annotated_image=annotated_b64,
    )

    return JSONResponse(content=response.model_dump())
