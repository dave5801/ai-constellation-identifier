from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from skimage.transform import AffineTransform

from vision.catalog import load_star_catalog
from vision.catalog_projection import catalog_star_magnitudes, project_catalog_stars
from vision.catalog_matching import (
    magnitude_to_relative_flux,
    match_catalog_entry,
    score_brightness_consistency,
)
from vision.constellation_match import ConstellationMatcher


def synthetic_stars_from_catalog(
    catalog_entry: dict,
    *,
    scale: float = 120.0,
    rotation_radians: float = 0.2,
    translation: tuple[float, float] = (260.0, 210.0),
    noise: float = 0.0,
) -> list[dict[str, float]]:
    source_points = project_catalog_stars(catalog_entry)
    transform = AffineTransform(
        scale=(scale, scale),
        rotation=rotation_radians,
        translation=translation,
    )
    transformed = transform(source_points)
    if noise > 0:
        transformed = transformed + np.full_like(transformed, noise)

    magnitudes = catalog_star_magnitudes(catalog_entry)
    return [
        {
            "x": float(point[0]),
            "y": float(point[1]),
            "brightness": float(10 ** (-0.4 * magnitude)),
            "confidence": 1.0,
        }
        for point, magnitude in zip(transformed, magnitudes, strict=False)
    ]


class CatalogMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog_path = Path(__file__).resolve().parent.parent / "data" / "star_catalog.json"
        cls.catalog = load_star_catalog(catalog_path)
        cls.matcher = ConstellationMatcher(cls.catalog)

    def test_every_catalog_entry_matches_synthetic_transform(self) -> None:
        for catalog_entry in self.catalog:
            stars = synthetic_stars_from_catalog(catalog_entry)
            match = match_catalog_entry(stars, catalog_entry)
            self.assertIsNotNone(match, msg=f"No direct match for {catalog_entry['name']}")
            assert match is not None
            self.assertEqual(match.name, catalog_entry["name"])

    def test_partial_pattern_does_not_match_large_constellation(self) -> None:
        orion = next(entry for entry in self.catalog if entry["name"] == "Orion")
        full_stars = synthetic_stars_from_catalog(orion)
        partial_stars = full_stars[:4]

        full_match = match_catalog_entry(full_stars, orion)
        partial_match = match_catalog_entry(partial_stars, orion)

        self.assertIsNotNone(full_match)
        self.assertIsNotNone(partial_match)
        assert full_match is not None
        assert partial_match is not None
        self.assertLess(partial_match.confidence, full_match.confidence)

    def test_brightness_consistency_scores_source_above_distractors(self) -> None:
        cygnus = next(entry for entry in self.catalog if entry["name"] == "Cygnus")
        magnitudes = catalog_star_magnitudes(cygnus)
        matched_indices = np.arange(len(magnitudes), dtype=np.int32)

        ideal_brightness = magnitude_to_relative_flux(magnitudes)
        mismatched_brightness = ideal_brightness[::-1]

        ideal_score = score_brightness_consistency(
            magnitudes,
            ideal_brightness,
            matched_indices,
            matched_indices,
        )
        mismatched_score = score_brightness_consistency(
            magnitudes,
            mismatched_brightness,
            matched_indices,
            matched_indices,
        )

        self.assertGreater(ideal_score, mismatched_score)

    def test_cluster_level_matching_finds_constellation_among_distractors(self) -> None:
        orion = next(entry for entry in self.catalog if entry["name"] == "Orion")
        stars = synthetic_stars_from_catalog(orion, translation=(180.0, 180.0))
        distractors = [
            {"x": 760.0, "y": 120.0, "brightness": 0.85, "confidence": 0.7},
            {"x": 790.0, "y": 150.0, "brightness": 0.82, "confidence": 0.7},
            {"x": 820.0, "y": 130.0, "brightness": 0.78, "confidence": 0.7},
            {"x": 770.0, "y": 90.0, "brightness": 0.76, "confidence": 0.7},
        ]

        matches = self.matcher.match(stars + distractors, width=1024, height=768)
        self.assertTrue(matches)
        self.assertIn("Orion", [match.name for match in matches])

    def test_dense_bright_field_without_pattern_does_not_accept_constellation(self) -> None:
        star_points = [
            (84.0, 96.0),
            (132.0, 114.0),
            (176.0, 92.0),
            (220.0, 148.0),
            (118.0, 186.0),
            (196.0, 214.0),
            (258.0, 192.0),
            (302.0, 136.0),
        ]
        stars = [
            {
                "x": x,
                "y": y,
                "brightness": 0.9 - (index * 0.03),
                "confidence": 0.8,
            }
            for index, (x, y) in enumerate(star_points)
        ]
        matches = self.matcher.match(stars, width=640, height=480)
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
