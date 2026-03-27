from __future__ import annotations

import asyncio
import io
import unittest

from PIL import Image, ImageDraw
from starlette.datastructures import Headers, UploadFile

from main import identify


class DebugApiTests(unittest.TestCase):
    @staticmethod
    def make_synthetic_image() -> bytes:
        image = Image.new("RGB", (192, 192), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        star_positions = [
            (30, 40),
            (62, 52),
            (96, 78),
            (132, 64),
            (148, 122),
            (78, 136),
            (42, 108),
        ]
        for x, y in star_positions:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_debug_response_contains_match_diagnostics(self) -> None:
        image_bytes = self.make_synthetic_image()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="synthetic.png",
            headers=Headers({"content-type": "image/png"}),
        )
        response = asyncio.run(identify(file=upload, debug=True))
        payload = response.body.decode("utf-8")
        import json
        data = json.loads(payload)
        self.assertIn("debug", data)
        self.assertIsInstance(data["debug"]["detected_stars"], list)
        self.assertIsInstance(data["debug"]["catalog_scores"], list)
        if data["debug"]["catalog_scores"]:
            first_score = data["debug"]["catalog_scores"][0]
            self.assertIn("cluster_id", first_score)
            self.assertIn("matched_star_count", first_score)
            self.assertIn("geometric_score", first_score)
            self.assertIn("coverage_score", first_score)
            self.assertIn("brightness_score", first_score)
            self.assertIn("compatibility_score", first_score)
            self.assertIn("rejection_reason", first_score)


if __name__ == "__main__":
    unittest.main()
