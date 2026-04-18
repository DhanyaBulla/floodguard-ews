import os
import pandas as pd
import numpy as np
import geopandas as gpd
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ── Load canal infrastructure layer ───────────────────────────
canals = gpd.read_file(
    os.path.join(DATA_DIR, 'canal_segments_1800.geojson'))
canals = canals.to_crs('EPSG:4326')

# No ML scoring for unmonitored canals
# These are infrastructure context only
canals['alert_tier']    = 'UNMONITORED'
canals['V_score_final'] = None
canals['display_color'] = '#888888'  # gray on map

print(f"Canal infrastructure segments: {len(canals)}")

# ── Load breach segments with real scores ──────────────────────
breach_scored = pd.read_csv(
    os.path.join(DATA_DIR, 'features_scored_v3.csv'))
breach_pts = breach_scored[breach_scored['label'] == 1].copy()

breach_map = gpd.GeoDataFrame(
    breach_pts,
    geometry=gpd.points_from_xy(
        breach_pts['centroid_lon'],
        breach_pts['centroid_lat']
    ),
    crs='EPSG:4326'
)

# Assign colors
def get_color(tier):
    return {'RED': '#d73027',
            'AMBER': '#fc8d59',
            'GREEN': '#1a9850'}.get(tier, '#888888')

breach_map['display_color'] = breach_map['alert_tier'].apply(get_color)

print(f"Breach segments scored    : {len(breach_map)}")
print(f"Breach tiers:\n{breach_map['alert_tier'].value_counts()}")
print(f"\nTop 5 highest risk:")
print(breach_map.nlargest(5,'V_score_final')[
    ['segment_id','study_area','V_score_final',
     'alert_tier','OT_risk','SP_risk','SL_risk']
].to_string())

# ── Round numerics ─────────────────────────────────────────────
for col in ['V_score_final','OT_risk','SP_risk','SL_risk']:
    if col in breach_map.columns:
        breach_map[col] = breach_map[col].round(3)

# ── Export ─────────────────────────────────────────────────────
canals.to_file(
    os.path.join(DATA_DIR, 'canals_scored.geojson'),
    driver='GeoJSON')
breach_map.to_file(
    os.path.join(DATA_DIR, 'breach_scored.geojson'),
    driver='GeoJSON')

print(f"\nSaved both files. Ready for dashboard.")