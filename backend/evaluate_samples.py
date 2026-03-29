from __future__ import annotations

import json
from pathlib import Path

import cv2

from vision.catalog import load_star_catalog
from vision.constellation_match import ConstellationMatcher
from vision.preprocess import preprocess_image
from vision.star_detection import detect_stars


def evaluate_sample_images(limit: int | None = None) -> dict[str, object]:
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    sample_manifest = json.loads((backend_dir / "data" / "sample_expectations.json").read_text(encoding="utf-8"))
    if limit is not None:
        sample_manifest = sample_manifest[:limit]
    matcher = ConstellationMatcher(load_star_catalog(backend_dir / "data" / "star_catalog.json"))

    report: list[dict] = []
    for item in sample_manifest:
        image_path = project_root / "sample_images" / item["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            report.append(
                {
                    "image": item["image"],
                    "expected_constellation": item["expected_constellation"],
                    "error": "image_not_found",
                }
            )
            continue

        processed = preprocess_image(image)
        stars = detect_stars(processed["grayscale"])
        matches = matcher.match(stars, image.shape[1], image.shape[0])
        evaluations = matcher.evaluate(stars, image.shape[1], image.shape[0])[:5]

        match_names = [match.name for match in matches]
        top_prediction = evaluations[0].name if evaluations else None
        report.append(
            {
                "image": item["image"],
                "expected_constellation": item["expected_constellation"],
                "stars_detected": len(stars),
                "accepted_matches": [
                    {"name": match.name, "confidence": round(match.confidence, 3)}
                    for match in matches
                ],
                "top_catalog_scores": [
                    {
                        "name": evaluation.name,
                        "confidence": round(evaluation.confidence, 3),
                        "accepted": evaluation.accepted,
                        "cluster_id": evaluation.cluster_id,
                        "rejection_reason": evaluation.rejection_reason,
                    }
                    for evaluation in evaluations
                ],
                "top_prediction": top_prediction,
                "expected_in_top_matches": item["expected_constellation"] in match_names,
                "expected_is_top_prediction": item["expected_constellation"] == top_prediction,
            }
        )

    successes = sum(1 for entry in report if entry.get("expected_in_top_matches"))
    top_prediction_hits = sum(1 for entry in report if entry.get("expected_is_top_prediction"))
    confusion_counts: dict[str, int] = {}
    for entry in report:
        expected = entry.get("expected_constellation")
        top_prediction = entry.get("top_prediction")
        if expected and top_prediction and expected != top_prediction:
            confusion_key = f"{expected} -> {top_prediction}"
            confusion_counts[confusion_key] = confusion_counts.get(confusion_key, 0) + 1

    summary = {
        "sample_count": len(report),
        "match_hit_count": successes,
        "match_hit_rate": round(successes / max(len(report), 1), 3),
        "top_prediction_hit_count": top_prediction_hits,
        "top_prediction_hit_rate": round(top_prediction_hits / max(len(report), 1), 3),
        "most_common_confusions": [
            {"pair": pair, "count": count}
            for pair, count in sorted(confusion_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        ],
    }

    return {"summary": summary, "samples": report}


if __name__ == "__main__":
    print(json.dumps(evaluate_sample_images(), indent=2))
