from __future__ import annotations

import unittest
from pathlib import Path

from vision.catalog import load_star_catalog


class StarCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog_path = Path(__file__).resolve().parent.parent / "data" / "star_catalog.json"
        cls.catalog = load_star_catalog(catalog_path)

    def test_catalog_contains_curated_expansion(self) -> None:
        self.assertGreaterEqual(len(self.catalog), 16)

    def test_each_constellation_has_valid_connections(self) -> None:
        for constellation in self.catalog:
            star_count = len(constellation["stars"])
            for connection in constellation["connections"]:
                self.assertEqual(len(connection), 2)
                self.assertTrue(0 <= connection[0] < star_count)
                self.assertTrue(0 <= connection[1] < star_count)

    def test_star_fields_parse_as_numeric(self) -> None:
        for constellation in self.catalog:
            for star in constellation["stars"]:
                self.assertIsInstance(float(star["ra_deg"]), float)
                self.assertIsInstance(float(star["dec_deg"]), float)
                self.assertIsInstance(float(star["mag"]), float)


if __name__ == "__main__":
    unittest.main()
