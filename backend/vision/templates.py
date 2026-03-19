from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_constellation_templates(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        templates = json.load(file)

    if not isinstance(templates, list):
        raise ValueError("Constellation templates must be a list.")

    for template in templates:
        if "name" not in template or "points" not in template or "connections" not in template:
            raise ValueError("Each constellation template must include name, points, and connections.")

    return templates
