from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def preprocess_image(image: np.ndarray) -> dict[str, np.ndarray]:
    """Prepare an RGB/BGR night-sky image for blob-based star detection."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(blurred)

    sharpened = cv2.addWeighted(contrast, 1.3, cv2.GaussianBlur(contrast, (0, 0), 3), -0.3, 0)
    top_hat = ndimage.white_tophat(sharpened, size=7)
    normalized = cv2.normalize(top_hat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    threshold_value = max(140, int(np.percentile(normalized, 98)))
    _, thresholded = cv2.threshold(normalized, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)

    return {
        "grayscale": grayscale,
        "contrast": contrast,
        "normalized": normalized,
        "thresholded": cleaned,
    }
