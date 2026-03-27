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
from api_models import CatalogScoreResponse, ConstellationResponse, DebugInfoResponse, IdentifyResponse
from vision.constellation_match import ConstellationMatcher
from vision.catalog import load_star_catalog
from vision.models import DetectionResult
from vision.preprocess import preprocess_image
from vision.star_detection import detect_stars, find_possible_planets

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
async def identify(file: UploadFile = File(...), debug: bool = False) -> JSONResponse:
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
    debug_detection = detect_stars(processed["grayscale"], debug=debug)
    if debug:
        assert isinstance(debug_detection, dict)
        stars = debug_detection["stars"]
    else:
        assert isinstance(debug_detection, list)
        stars = debug_detection
    detection = DetectionResult(
        stars=stars,
        possible_planets=find_possible_planets(stars),
    )
    evaluations = matcher.evaluate(detection.stars, image_array.shape[1], image_array.shape[0]) if debug else []
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
        debug=(
            DebugInfoResponse(
                detected_stars=[
                    {
                        "x": round(star["x"], 2),
                        "y": round(star["y"], 2),
                        "brightness": round(star["brightness"], 4),
                        "confidence": round(star.get("confidence", 0.0), 4),
                    }
                    for star in detection.stars
                ],
                raw_candidates=[
                    {
                        "x": round(candidate["x"], 2),
                        "y": round(candidate["y"], 2),
                        "brightness": round(candidate["brightness"], 4),
                        "confidence": round(candidate.get("confidence", 0.0), 4),
                    }
                    for candidate in debug_detection["raw_candidates"]
                ],
                filtered_candidates=[
                    {
                        "x": round(candidate["x"], 2),
                        "y": round(candidate["y"], 2),
                        "brightness": round(candidate["brightness"], 4),
                        "confidence": round(candidate.get("confidence", 0.0), 4),
                    }
                    for candidate in debug_detection["filtered_candidates"]
                ],
                catalog_scores=[
                    CatalogScoreResponse(
                        name=evaluation.name,
                        confidence=round(evaluation.confidence, 3),
                        accepted=evaluation.accepted,
                        cluster_id=evaluation.cluster_id,
                        matched_star_count=evaluation.matched_star_count,
                        geometric_score=round(evaluation.geometric_score, 3),
                        coverage_score=round(evaluation.coverage_score, 3),
                        brightness_score=round(evaluation.brightness_score, 3),
                        compatibility_score=round(evaluation.compatibility_score, 3),
                        rejection_reason=evaluation.rejection_reason,
                    )
                    for evaluation in evaluations
                ],
            )
            if debug
            else None
        ),
    )

    return JSONResponse(content=response.model_dump())
