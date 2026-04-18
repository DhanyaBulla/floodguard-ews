import os
import pandas as pd
import numpy as np
import geopandas as gpd
import pickle
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

model    = pickle.load(open(os.path.join(DATA_DIR, 'model.pkl'),         'rb'))
imputer  = pickle.load(open(os.path.join(DATA_DIR, 'imputer.pkl'),       'rb'))
features = pickle.load(open(os.path.join(DATA_DIR, 'feature_names.pkl'), 'rb'))

train_df        = pd.read_csv(os.path.join(DATA_DIR, 'features_scored_v3.csv'))
breach_df       = train_df[train_df['label'] == 1]
control_df      = train_df[train_df['label'] == 0]
breach_medians  = breach_df[features].median()
control_medians = control_df[features].median()

canals = gpd.read_file(
    os.path.join(DATA_DIR, 'canal_segments_1800.geojson'))
canals = canals.to_crs('EPSG:4326')
print(f"Canal segments: {len(canals)}")

# ── District from coordinates ──────────────────────────────────
def get_district(lon, lat):
    if 80.2 <= lon <= 81.5 and 15.5 <= lat <= 17.5:
        return 'Krishna'
    elif 81.5 <= lon <= 82.5 and 16.0 <= lat <= 17.5:
        return 'East Godavari'
    elif 80.8 <= lon <= 81.8 and 16.5 <= lat <= 17.5:
        return 'West Godavari'
    elif 80.5 <= lon <= 81.5 and 15.5 <= lat <= 16.5:
        return 'Guntur'
    elif 79.0 <= lon <= 80.5 and 15.0 <= lat <= 16.5:
        return 'Y.S.R.Kadapa'
    elif 77.5 <= lon <= 79.5 and 14.5 <= lat <= 16.0:
        return 'Kurnool'
    elif 77.0 <= lon <= 78.5 and 13.5 <= lat <= 15.0:
        return 'Anantapur'
    elif 79.0 <= lon <= 80.5 and 13.0 <= lat <= 14.5:
        return 'Tirupati'
    elif 79.5 <= lon <= 81.0 and 14.0 <= lat <= 15.5:
        return 'S.P.S.Nellore'
    elif 83.0 <= lon <= 84.5 and 17.5 <= lat <= 19.0:
        return 'Srikakulam'
    elif 83.0 <= lon <= 84.0 and 17.0 <= lat <= 18.5:
        return 'Vizianagaram'
    elif 82.0 <= lon <= 83.5 and 17.0 <= lat <= 18.5:
        return 'Visakhapatnam'
    elif 81.0 <= lon <= 82.5 and 16.0 <= lat <= 17.0:
        return 'Eluru'
    elif 81.5 <= lon <= 82.5 and 15.5 <= lat <= 16.5:
        return 'Konaseema'
    else:
        return 'Unknown'

# ── Compressed district vulnerability ─────────────────────────
district_vuln = {
    'Krishna':        0.72,
    'East Godavari':  0.68,
    'West Godavari':  0.65,
    'Konaseema':      0.63,
    'Guntur':         0.60,
    'Eluru':          0.55,
    'Srikakulam':     0.52,
    'Y.S.R.Kadapa':   0.55,
    'Vizianagaram':   0.48,
    'Visakhapatnam':  0.45,
    'S.P.S.Nellore':  0.48,
    'Tirupati':       0.50,
    'Kurnool':        0.42,
    'Anantapur':      0.35,
    'Unknown':        0.45,
}

canal_vuln = {
    'Main Canal':   0.75,
    'Major Canal':  0.68,
    'Branch Canal': 0.55,
}

np.random.seed(42)
canal_feats        = []
vuln_scores        = []
districts_assigned = []

for _, row in canals.iterrows():
    lon = float(row.get('centroid_lon', 0) or 0)
    lat = float(row.get('centroid_lat', 0) or 0)
    if lon == 0 or lat == 0:
        c = row.geometry.centroid
        lon, lat = c.x, c.y

    district = str(row.get('district', '') or '')
    if not district or district in ('nan', ''):
        district = get_district(lon, lat)
    districts_assigned.append(district)

    ctype    = str(row.get('canal_type', 'Branch Canal'))
    d_vuln   = district_vuln.get(district, 0.45)
    c_vuln   = canal_vuln.get(ctype, 0.55)
    combined = 0.60 * d_vuln + 0.40 * c_vuln

    # Interpolate between control and breach medians
    # using district vulnerability as alpha
    # alpha=0 → control conditions, alpha=1 → breach conditions
    alpha = d_vuln
    feat  = {}
    for col in features:
        ctrl_val   = control_medians[col]
        breach_val = breach_medians[col]
        interp     = ctrl_val + alpha * (breach_val - ctrl_val)
        noise      = np.random.normal(
            0, abs(interp) * 0.10 if interp != 0 else 0.01)
        feat[col]  = interp + noise

    feat['seg_length_m'] = float(row.get('seg_length_m', 200))
    canal_feats.append(feat)
    vuln_scores.append({
        'd_vuln':   d_vuln,
        'c_vuln':   c_vuln,
        'combined': combined
    })

canals['district_assigned'] = districts_assigned
cf  = pd.DataFrame(canal_feats)[features]
vdf = pd.DataFrame(vuln_scores)

print(f"\nDistrict assignment:")
print(pd.Series(districts_assigned).value_counts())

# ── ML score ───────────────────────────────────────────────────
X_scored         = imputer.transform(cf)
canals['V_score_ml'] = model.predict_proba(X_scored)[:, 1]

# ── Physical sub-scores ────────────────────────────────────────
def minmax(s):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

OT = (minmax(cf['NDWI_anomaly'])     * 0.35 +
      minmax(cf['S1_flood_proxy'])   * 0.35 +
      minmax(cf['LSWI_anomaly'])     * 0.30)

SP = (minmax(cf['smap_sm_mean'])     * 0.25 +
      minmax(cf['era5_sm_l2'])        * 0.25 +
      minmax(cf['sm_days_above_fc'])  * 0.30 +
      minmax(cf['S1_SMI'])            * 0.20)

SL = (minmax(cf['TWI'])              * 0.40 +
      minmax(cf['slope'])             * 0.30 +
      minmax(-cf['NDVI_anomaly'])     * 0.30)

canals['OT_risk']          = OT.values
canals['SP_risk']          = SP.values
canals['SL_risk']          = SL.values
canals['V_score_physical'] = (0.40*OT + 0.35*SP + 0.25*SL).values
canals['district_vuln']    = vdf['d_vuln'].values

# ── Final blended score ────────────────────────────────────────
# Reduced district weight to 0.20 to prevent domination
canals['V_score_final'] = (
    0.40 * canals['V_score_ml'] +
    0.40 * canals['V_score_physical'] +
    0.20 * vdf['combined'].values
)

# ── Alert tiers ────────────────────────────────────────────────
def assign_tier(row):
    v  = row['V_score_final']
    dv = row['district_vuln']
    sp = row['SP_risk']
    if v > 0.65:
        return 'RED'
    elif v > 0.50 or (dv >= 0.65 and v > 0.45):
        return 'AMBER'
    else:
        return 'GREEN'

canals['alert_tier'] = canals.apply(assign_tier, axis=1)

# ── Dominant failure mode ──────────────────────────────────────
def dominant(row):
    scores = {
        'Overtopping':       row['OT_risk'],
        'Seepage/Piping':    row['SP_risk'],
        'Slope Instability': row['SL_risk'],
    }
    return max(scores, key=scores.get)

canals['dominant_mode'] = canals.apply(dominant, axis=1)

# ── Summary ────────────────────────────────────────────────────
print(f"\nAlert tier distribution:")
print(canals['alert_tier'].value_counts())

print(f"\nV_score distribution:")
print(canals['V_score_final'].describe().round(3))

print(f"\nBy district and tier:")
pivot = canals.groupby(
    ['district_assigned','alert_tier'])['segment_id'] \
    .count().unstack(fill_value=0)
print(pivot)

print(f"\nTop 10 highest risk canals:")
print(canals.nlargest(10, 'V_score_final')[[
    'parent_name','canal_type','district_assigned',
    'V_score_final','alert_tier','dominant_mode'
]].to_string())

# ── Breach overlay ─────────────────────────────────────────────
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

if 'alert_tier' not in breach_map.columns:
    def tier(row):
        v = row['V_score_final']
        if v > 0.65:   return 'RED'
        elif v > 0.40: return 'AMBER'
        else:          return 'GREEN'
    breach_map['alert_tier'] = breach_map.apply(tier, axis=1)

print(f"\nBreach overlay: {len(breach_map)} segments")
print(f"Breach tiers:\n{breach_map['alert_tier'].value_counts()}")

# ── Round numerics ─────────────────────────────────────────────
num_cols = ['V_score_final','V_score_physical','V_score_ml',
            'OT_risk','SP_risk','SL_risk','district_vuln']
for col in num_cols:
    for gdf in [canals, breach_map]:
        if col in gdf.columns:
            gdf[col] = gdf[col].round(3)

# ── Export ─────────────────────────────────────────────────────
canals.to_file(
    os.path.join(DATA_DIR, 'canals_scored.geojson'),
    driver='GeoJSON')
breach_map.to_file(
    os.path.join(DATA_DIR, 'breach_scored.geojson'),
    driver='GeoJSON')

print(f"\n── Saved ──")
print(f"  canals_scored.geojson : {len(canals)}")
print(f"  breach_scored.geojson : {len(breach_map)}")
print("\nDone. Ready for dashboard.")