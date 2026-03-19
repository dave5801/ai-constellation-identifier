from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from vision.constellation_match import ConstellationMatcher, MatchResult
from vision.models import DetectionResult
from vision.preprocess import preprocess_image
from vision.star_detection import detect_stars
from vision.templates import load_constellation_templates


class ConstellationResponse(BaseModel):
    name: str
    confidence: float


class IdentifyResponse(BaseModel):
    constellations: list[ConstellationResponse]
    stars_detected: int
    possible_planets: list[dict[str, float]]
    annotated_image: str


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "data" / "constellation_templates.json"

matcher = ConstellationMatcher(load_constellation_templates(TEMPLATE_PATH))

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


def _encode_image(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _draw_constellation_outline(
    image: np.ndarray,
    match: MatchResult,
    color: tuple[int, int, int],
) -> None:
    points = np.array(match.transformed_points, dtype=np.float32)
    if len(points) < 3:
        return

    hull = cv2.convexHull(points.astype(np.int32))
    cv2.polylines(image, [hull], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    x, y, width, height = cv2.boundingRect(hull)
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_annotations(
    image: np.ndarray,
    detection: DetectionResult,
    matches: list[MatchResult],
) -> np.ndarray:
    annotated = image.copy()

    for star in detection.stars:
        center = (int(round(star["x"])), int(round(star["y"])))
        radius = max(3, int(round(star["size"] / 2)))
        cv2.circle(annotated, center, radius, (0, 255, 255), 1)

    for planet in detection.possible_planets:
        center = (int(round(planet["x"])), int(round(planet["y"])))
        cv2.circle(annotated, center, 10, (255, 140, 0), 2)
        cv2.putText(
            annotated,
            "Possible planet",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 140, 0),
            1,
            cv2.LINE_AA,
        )

    for match in matches:
        color = tuple(int(channel) for channel in match.color)
        _draw_constellation_outline(annotated, match, color)

        for start_idx, end_idx in match.connections:
            start = tuple(int(round(value)) for value in match.transformed_points[start_idx])
            end = tuple(int(round(value)) for value in match.transformed_points[end_idx])
            cv2.line(annotated, start, end, color, 2, cv2.LINE_AA)

        anchor = tuple(int(round(value)) for value in match.transformed_points[0])
        label = f"{match.name} ({match.confidence:.2f})"
        cv2.putText(
            annotated,
            label,
            (anchor[0] + 6, anchor[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


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
    annotated = _draw_annotations(image_array, detection, matches)
    annotated_b64 = _encode_image(annotated)

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
