from __future__ import annotations

from vision.catalog_matching import match_catalog_entry
from vision.models import MatchResult, Star


class ConstellationMatcher:
    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog

    def match(self, stars: list[Star], width: int, height: int) -> list[MatchResult]:
        del width, height

        results = []
        for catalog_entry in self.catalog:
            match = match_catalog_entry(stars, catalog_entry)
            if match:
                results.append(match)

        results.sort(key=lambda item: item.confidence, reverse=True)
        return results[:3]
