from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_star_catalog(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)

    if not isinstance(catalog, list):
        raise ValueError("Star catalog must be a list.")

    for constellation in catalog:
        if "name" not in constellation or "stars" not in constellation or "connections" not in constellation:
            raise ValueError("Each catalog entry must include name, stars, and connections.")

        for star in constellation["stars"]:
            if "ra_deg" not in star or "dec_deg" not in star:
                raise ValueError("Each catalog star must include ra_deg and dec_deg.")

    return catalog
