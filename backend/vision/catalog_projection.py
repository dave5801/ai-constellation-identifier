from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


def project_catalog_stars(catalog_entry: dict) -> np.ndarray:
    sky_coords = SkyCoord(
        ra=[star["ra_deg"] for star in catalog_entry["stars"]] * u.deg,
        dec=[star["dec_deg"] for star in catalog_entry["stars"]] * u.deg,
        frame="icrs",
    )
    wrapped_ra = sky_coords.ra.wrap_at(180 * u.deg)
    center_ra = np.mean(wrapped_ra.deg)
    center_dec = np.mean(sky_coords.dec.deg)
    delta_ra = wrapped_ra.deg - center_ra
    cos_dec = np.cos(np.deg2rad(center_dec))
    x = delta_ra * cos_dec
    y = sky_coords.dec.deg - center_dec
    return np.column_stack([x, y]).astype(np.float32)


def catalog_star_magnitudes(catalog_entry: dict) -> np.ndarray:
    return np.array([star["mag"] for star in catalog_entry["stars"]], dtype=np.float32)
