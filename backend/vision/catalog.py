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
        if not isinstance(constellation["stars"], list) or len(constellation["stars"]) < 3:
            raise ValueError("Each catalog entry must include at least three stars.")
        if not isinstance(constellation["connections"], list):
            raise ValueError("Each catalog entry must include a list of connections.")

        for star in constellation["stars"]:
            if "ra_deg" not in star or "dec_deg" not in star or "mag" not in star:
                raise ValueError("Each catalog star must include ra_deg, dec_deg, and mag.")
            float(star["ra_deg"])
            float(star["dec_deg"])
            float(star["mag"])

        star_count = len(constellation["stars"])
        for connection in constellation["connections"]:
            if len(connection) != 2:
                raise ValueError("Each connection must contain exactly two star indices.")
            start_idx, end_idx = connection
            if not (0 <= int(start_idx) < star_count and 0 <= int(end_idx) < star_count):
                raise ValueError("Connection indices must reference valid stars.")

    return catalog
