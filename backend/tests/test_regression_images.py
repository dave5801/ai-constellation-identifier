from __future__ import annotations

import unittest
import json
from pathlib import Path

import cv2

from evaluate_samples import evaluate_sample_images
from vision.catalog import load_star_catalog
from vision.constellation_match import ConstellationMatcher
from vision.preprocess import preprocess_image
from vision.star_detection import detect_stars


class RegressionImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        backend_dir = Path(__file__).resolve().parent.parent
        project_root = backend_dir.parent
        cls.matcher = ConstellationMatcher(load_star_catalog(backend_dir / "data" / "star_catalog.json"))
        cls.sample_dir = project_root / "sample_images"
        cls.sample_manifest_path = backend_dir / "data" / "sample_expectations.json"

    def run_image_pipeline(self, image_name: str) -> tuple[list[dict[str, float]], list[str]]:
        image_path = self.sample_dir / image_name
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image, msg=f"Could not load sample image {image_name}")

        processed = preprocess_image(image)
        stars = detect_stars(processed["grayscale"])
        matches = self.matcher.match(stars, image.shape[1], image.shape[0])
        return stars, [match.name for match in matches]

    def test_orion_sample_runs_detection_pipeline(self) -> None:
        stars, match_names = self.run_image_pipeline("orion.png")
        self.assertGreaterEqual(len(stars), 5)
        self.assertIsInstance(match_names, list)

    def test_cygnus_sample_runs_detection_pipeline(self) -> None:
        stars, match_names = self.run_image_pipeline("cygnus.png")
        self.assertGreaterEqual(len(stars), 5)
        self.assertIsInstance(match_names, list)

    def test_sample_manifest_is_well_formed(self) -> None:
        manifest = json.loads(self.sample_manifest_path.read_text(encoding="utf-8"))
        self.assertIsInstance(manifest, list)
        self.assertGreaterEqual(len(manifest), 2)
        for entry in manifest:
            self.assertIn("image", entry)
            self.assertIn("expected_constellation", entry)

    def test_sample_evaluation_workflow_returns_report_entries(self) -> None:
        report = evaluate_sample_images(limit=1)
        self.assertEqual(len(report), 1)
        for entry in report:
            self.assertIn("image", entry)
            self.assertIn("expected_constellation", entry)
            self.assertIn("stars_detected", entry)
            self.assertIn("accepted_matches", entry)
            self.assertIn("top_catalog_scores", entry)
            self.assertIn("expected_in_top_matches", entry)


if __name__ == "__main__":
    unittest.main()
