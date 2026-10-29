"""Generate realistic synthetic 11-band Sentinel-2 coastal imagery for MARIDA U-Net testing.

Simulates authentic Bottom-Of-Atmosphere (BOA) reflectance across Sentinel-2 bands:
- Band 1: B1 (Coastal Aerosol, 443nm)
- Band 2: B2 (Blue, 490nm)
- Band 3: B3 (Green, 560nm)
- Band 4: B4 (Red, 665nm)
- Band 5: B5 (Vegetation Red Edge 1, 705nm)
- Band 6: B6 (Vegetation Red Edge 2, 740nm)
- Band 7: B7 (Vegetation Red Edge 3, 783nm)
- Band 8: B8 (NIR Broad, 842nm)
- Band 9: B8A (NIR Narrow, 865nm)
- Band 10: B11 (SWIR 1, 1610nm)
- Band 11: B12 (SWIR 2, 2190nm)

Key Spatial & Spectral Features:
1. Deep/Coastal Ocean: Low visible & near-infrared reflectance (smooth deep blue/cyan preview).
2. Coastline & Land: Sandy beach and vegetated boundary with characteristic spectral peaks.
3. Debris Slick: Concentrated cluster in open water with elevated NIR/SWIR polymer reflectance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def generate_accra_sample(height: int = 256, width: int = 256) -> np.ndarray:
    """Generate realistic 11-band multispectral coastal imagery."""
    y, x = np.mgrid[0:height, 0:width]
    rng = np.random.default_rng(42)

    # 1. Geographic Boundary: Accra Coast faces South onto the Gulf of Guinea.
    # Land is situated to the North (y < coast), open ocean to the South (y >= coast).
    coast = 55.0 + 12.0 * np.sin(x / 24.0) + 6.0 * np.cos(x / 11.0) + 2.0 * np.sin(x / 4.0)
    dist = y - coast  # Negative = land, positive = water

    # 2. Ocean Swell & Surface Waves (smooth natural marine texture, no TV-static noise)
    swell = 0.006 * np.sin(x / 14.0 + y / 9.0) + 0.004 * np.cos(x / 7.0 - y / 16.0)
    ripples = 0.002 * np.sin(x * 0.9 + y * 0.7)

    # Initialize 11 multispectral bands
    data = np.zeros((11, height, width), dtype=np.float32)

    # -------------------------------------------------------------------------
    # Deep & Coastal Ocean (low visible/NIR reflectance, smooth blue/cyan in RGB)
    # -------------------------------------------------------------------------
    ocean_scale = 0.25
    ocean_mask = dist >= 0

    data[0, ocean_mask] = ocean_scale * 1.05 + swell[ocean_mask]
    data[1, ocean_mask] = ocean_scale * 1.00 + swell[ocean_mask] + ripples[ocean_mask]  # B2 (Blue)
    data[2, ocean_mask] = ocean_scale * 0.85 + swell[ocean_mask] * 0.8                # B3 (Green)
    data[3, ocean_mask] = ocean_scale * 0.38 + swell[ocean_mask] * 0.4                # B4 (Red)
    data[4, ocean_mask] = ocean_scale * 0.30                                          # B5 (Red Edge 1)
    data[5, ocean_mask] = ocean_scale * 0.25                                          # B6 (Red Edge 2)
    data[6, ocean_mask] = ocean_scale * 0.25                                          # B7 (Red Edge 3)
    data[7, ocean_mask] = ocean_scale * 0.20                                          # B8 (NIR Broad)
    data[8, ocean_mask] = ocean_scale * 0.20                                          # B8A (NIR Narrow)
    data[9, ocean_mask] = ocean_scale * 0.15                                          # B11 (SWIR 1)
    data[10, ocean_mask] = ocean_scale * 0.10                                         # B12 (SWIR 2)

    # Nearshore shallow water gradient (sediment & depth transition, 0 <= dist < 22)
    shallow_mask = (dist >= 0) & (dist < 22)
    t_shallow = 1.0 - (dist[shallow_mask] / 22.0)
    data[1, shallow_mask] += 0.03 * t_shallow
    data[2, shallow_mask] += 0.04 * t_shallow
    data[3, shallow_mask] += 0.02 * t_shallow

    # -------------------------------------------------------------------------
    # Coastline: Sandy Shoreline / Beach (0 <= dist < 8)
    # High reflectance across visible spectrum, warm golden sand tone
    # -------------------------------------------------------------------------
    beach_mask = (dist >= 0) & (dist < 8)
    t_beach = 1.0 - (dist[beach_mask] / 8.0)
    data[1, beach_mask] += 0.10 * t_beach  # B2 Blue
    data[2, beach_mask] += 0.14 * t_beach  # B3 Green
    data[3, beach_mask] += 0.18 * t_beach  # B4 Red
    data[7, beach_mask] += 0.16 * t_beach  # B8 NIR
    data[9, beach_mask] += 0.12 * t_beach  # B11 SWIR1

    # -------------------------------------------------------------------------
    # Coastline: Inland Vegetation & Coastal Terrain (dist < 0)
    # Low Red (chlorophyll absorption), Green peak, very high NIR reflectance
    # -------------------------------------------------------------------------
    land_mask = dist < 0
    land_texture = 0.015 * np.sin(x / 5.0) * np.cos(y / 5.0)

    data[0, land_mask] = 0.14
    data[1, land_mask] = 0.17 + land_texture[land_mask]  # B2 Blue (low)
    data[2, land_mask] = 0.36 + land_texture[land_mask]  # B3 Green (lush vegetation)
    data[3, land_mask] = 0.20 + land_texture[land_mask]  # B4 Red (absorbed)
    data[4, land_mask] = 0.32
    data[5, land_mask] = 0.50
    data[6, land_mask] = 0.60
    data[7, land_mask] = 0.68 + land_texture[land_mask]  # B8 NIR (strong vegetation peak)
    data[8, land_mask] = 0.70
    data[9, land_mask] = 0.28
    data[10, land_mask] = 0.16

    # -------------------------------------------------------------------------
    # Debris Slick: Concentrated Floating Marine Plastic Cluster
    # Distinct polymer signature: elevated NIR (B8, B8A) & SWIR (B11, B12)
    # -------------------------------------------------------------------------
    slick_dist = ((y - 150) / 12.0) ** 2 + ((x - 130) / 25.0 + 0.3 * np.sin((y - 150) / 3.0)) ** 2
    noise_filter = rng.uniform(0.0, 1.0, (height, width)) > 0.15
    cluster = (slick_dist < 1.0) & noise_filter

    data[1, cluster] += 0.02  # B2 subtle visible filament contrast
    data[2, cluster] += 0.02  # B3
    data[3, cluster] += 0.03  # B4
    data[5, cluster] = 0.35   # B6 Red Edge
    data[7, cluster] = 0.35   # B8 NIR (elevated plastic reflection)
    data[8, cluster] = 0.35   # B8A NIR
    data[9, cluster] = 0.40   # B11 SWIR 1 (elevated plastic reflection)
    data[10, cluster] = 0.32  # B12 SWIR 2

    return np.clip(data, 0.0, 1.0).astype(np.float32)


def main() -> None:
    output_path = Path(__file__).parent / "samples" / "accra-coast.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_accra_sample(256, 256)

    # GeoTIFF metadata matching Accra, Ghana (-0.20° W, 5.55° N)
    profile = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 11,
        "dtype": "float32",
        "transform": rasterio.transform.from_origin(-0.20, 5.55, 10.0, 10.0),
        "crs": "EPSG:4326",
    }

    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(data)

    print(f"Successfully generated realistic coastal Sentinel-2 sample at {output_path}")


if __name__ == "__main__":
    main()