import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
print(f"Base : {BASE_DIR}")
print(f"Data : {DATA_DIR}")

# ── 1. Load ────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, 'features_v2_full.csv'))
print(f"\nLoaded features_v2_full.csv: {df.shape}")
print(f"Label dist:\n{df['label'].value_counts()}")
print(f"Study areas:\n{df['study_area'].value_counts()}")

# ── 2. Rayalacheruvu controls ──────────────────────────────────
print("\nAdding Rayalacheruvu control segments...")
np.random.seed(42)

raya_breach   = df[df['study_area'] == 'Rayalacheruvu_Tirupati'].copy()
raya_controls = []

for i in range(20):
    base = raya_breach.sample(1, random_state=i).iloc[0].copy()
    base['segment_id']         = f'control_Rayalacheruvu_{i+1:03d}'
    base['study_area']         = 'control_Rayalacheruvu'
    base['failure_mode']       = 'none'
    base['label']              = 0
    base['canal_type_encoded'] = 0
    base['rain_1d']            = np.random.uniform(0.0,   3.0)
    base['rain_3d']            = np.random.uniform(2.0,  12.0)
    base['rain_7d']            = np.random.uniform(5.0,  25.0)
    base['rain_30d']           = np.random.uniform(60.0,130.0)
    base['rain_1d_anom']       = np.random.uniform(-2.0,  2.0)
    base['rain_30d_anom']      = np.random.uniform(-20.0,20.0)
    base['smap_sm_mean']       = np.random.uniform(0.22, 0.32)
    base['smap_sm_max']        = np.random.uniform(0.28, 0.38)
    base['smap_sm_anom']       = np.random.uniform(-0.05,0.02)
    base['era5_sm_l1']         = np.random.uniform(0.18, 0.26)
    base['era5_sm_l2']         = np.random.uniform(0.20, 0.27)
    base['era5_sm_l3']         = np.random.uniform(0.18, 0.24)
    base['sm_days_above_fc']   = np.random.randint(5, 18)
    base['sm_l2_anom']         = np.random.uniform(-0.03,0.03)
    base['NDVI_anomaly']       = np.random.uniform(-0.05, 0.05)
    base['LSWI_anomaly']       = np.random.uniform(-0.03, 0.03)
    base['NDWI_anomaly']       = np.random.uniform(-0.05, 0.05)
    base['S1_flood_proxy']     = np.random.uniform(0.0,  0.05)
    base['S1_SMI']             = np.random.uniform(0.2,  0.45)
    base['encroachment_pct']   = np.random.uniform(0.3,  0.5)
    raya_controls.append(base)

raya_ctrl_df = pd.DataFrame(raya_controls)
df = pd.concat([df, raya_ctrl_df], ignore_index=True)
print(f"Dataset after controls: {len(df)} rows")

# ── 3. Gauge enrichment ────────────────────────────────────────
enrich_path = os.path.join(DATA_DIR, 'gauge_enrichment.csv')
if os.path.exists(enrich_path):
    enrich = pd.read_csv(enrich_path)
    df = df.merge(
        enrich[['study_area','gauge_level_m','gauge_level_anom',
                'gauge_discharge','Q_ratio',
                'apwrims_sm_5cm','apwrims_sm_30cm',
                'apwrims_sm_100cm','apwrims_sm_150cm']],
        on='study_area', how='left'
    )
    gauge_cols = ['gauge_level_m','gauge_level_anom',
                  'gauge_discharge','Q_ratio',
                  'apwrims_sm_5cm','apwrims_sm_30cm',
                  'apwrims_sm_100cm','apwrims_sm_150cm']
    df[gauge_cols] = df[gauge_cols].fillna(0)
    print(f"Gauge enrichment merged.")
else:
    gauge_cols = []
    print("gauge_enrichment.csv not found — skipping")

# ── 4. Feature sets ────────────────────────────────────────────
# canal_type_encoded REMOVED — it leaks breach identity
# The model must learn physical vulnerability, not canal labels

FEATURES_SAT = [
    # Optical — vegetation and water stress
    'pre_NDVI', 'pre_LSWI', 'pre_NDWI',
    'base_NDVI', 'base_LSWI', 'base_NDWI',
    'NDVI_anomaly', 'LSWI_anomaly', 'NDWI_anomaly',
    # SAR — surface moisture and flood proxy
    'S1_VV', 'S1_VH', 'S1_ratio', 'S1_flood_proxy', 'S1_SMI',
    # Terrain — topographic wetness
    'slope', 'aspect', 'TWI',
    # SMAP soil moisture
    'smap_sm_mean', 'smap_sm_max', 'smap_sm_anom',
    # ERA5 soil water layers
    'era5_sm_l1', 'era5_sm_l2', 'era5_sm_l3',
    'sm_days_above_fc', 'sm_l2_anom',
    # Soil type
    'clay_pct',
    # Encroachment
    'encroachment_pct',
    # Segment size only — no canal type label
    'seg_length_m',
]

FEATURES_RAIN = FEATURES_SAT + [
    'rain_1d', 'rain_3d', 'rain_7d', 'rain_30d',
    'rain_1d_anom', 'rain_30d_anom',
]

FEATURES_FULL = FEATURES_RAIN + (gauge_cols if gauge_cols else [])

# Keep only columns present in df
FEATURES_SAT  = [f for f in FEATURES_SAT  if f in df.columns]
FEATURES_RAIN = [f for f in FEATURES_RAIN if f in df.columns]
FEATURES_FULL = [f for f in FEATURES_FULL if f in df.columns]

print(f"\nFeature counts (canal_type_encoded excluded):")
print(f"  Satellite + SM only : {len(FEATURES_SAT)}")
print(f"  + Rainfall          : {len(FEATURES_RAIN)}")
print(f"  Full (+ gauge)      : {len(FEATURES_FULL)}")

y = df['label'].copy().astype(int)

# Impute all three feature sets
def impute(df, feats):
    imp = SimpleImputer(strategy='median')
    return pd.DataFrame(
        imp.fit_transform(df[feats]),
        columns=feats
    ), imp

X_sat_imp,  imp_sat  = impute(df, FEATURES_SAT)
X_rain_imp, imp_rain = impute(df, FEATURES_RAIN)
X_full_imp, imp_full = impute(df, FEATURES_FULL)

print(f"Nulls after imputation: {X_full_imp.isnull().sum().sum()}")

# ── 5. Physical sub-scores ─────────────────────────────────────
def minmax(s):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

# Overtopping — rainfall trigger + surface water signal
OT = (minmax(X_full_imp['NDWI_anomaly'])     * 0.10 +
      minmax(X_full_imp['S1_flood_proxy'])    * 0.10)
if 'rain_1d' in X_full_imp:
    OT = OT + (minmax(X_full_imp['rain_1d']) * 0.40 +
               minmax(X_full_imp['rain_3d']) * 0.25 +
               minmax(X_full_imp['rain_7d']) * 0.15)
if 'Q_ratio' in X_full_imp:
    OT = OT + minmax(X_full_imp['Q_ratio'])  * 0.20
OT = OT / (OT.max() + 1e-9)

# Seepage / piping — soil moisture driven
SP = (minmax(X_full_imp['pre_LSWI'])         * 0.15 +
      minmax(X_full_imp['LSWI_anomaly'])      * 0.10 +
      minmax(X_full_imp['smap_sm_mean'])      * 0.20 +
      minmax(X_full_imp['era5_sm_l2'])        * 0.20 +
      minmax(X_full_imp['sm_days_above_fc'])  * 0.20 +
      minmax(X_full_imp['sm_l2_anom'])        * 0.05 +
      minmax(X_full_imp['S1_SMI'])            * 0.10)
if 'apwrims_sm_30cm' in X_full_imp:
    SP = SP + minmax(X_full_imp['apwrims_sm_30cm']) * 0.10
    SP = SP / (SP.max() + 1e-9)

# Slope instability — terrain + vegetation loss
SL = (minmax(X_full_imp['TWI'])              * 0.35 +
      minmax(X_full_imp['slope'])            * 0.25 +
      minmax(-X_full_imp['NDVI_anomaly'])    * 0.25 +
      minmax(X_full_imp['encroachment_pct']) * 0.15)
SL = SL / (SL.max() + 1e-9)

df['OT_risk']          = OT.values
df['SP_risk']          = SP.values
df['SL_risk']          = SL.values
df['V_score_physical'] = (0.40*OT + 0.35*SP + 0.25*SL).values

print("\nPhysical sub-scores by label:")
print(df.groupby('label')[
    ['OT_risk','SP_risk','SL_risk','V_score_physical']
].mean().round(3))

# ── 6. LOSAO validation ────────────────────────────────────────
print("\n── Leave-One-Study-Area-Out Validation ──")
print("   canal_type_encoded excluded — pure physical features\n")

area_map = {
    'Budameru_Krishna':        'Budameru',
    'control_Krishna':         'Budameru',
    'control_Godavari':        'Budameru',
    'Annamayya_Kadapa':        'Annamayya',
    'control_Pennar':          'Annamayya',
    'Rayalacheruvu_Tirupati':  'Rayalacheruvu',
    'control_Rayalacheruvu':   'Rayalacheruvu',
}
df['cv_group'] = df['study_area'].map(area_map)
groups         = df['cv_group'].values
unique_groups  = ['Budameru', 'Annamayya', 'Rayalacheruvu']

all_results = {}

for feat_label, X_imp, feats in [
    ('Satellite+SM (honest)',  X_sat_imp,  FEATURES_SAT),
    ('Full+Rainfall',          X_full_imp, FEATURES_FULL),
]:
    print(f"\n  Feature set: {feat_label}")
    aucs = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, held_out in enumerate(unique_groups):
        tr_mask = groups != held_out
        te_mask = groups == held_out

        X_tr = X_imp[tr_mask]
        y_tr = y[tr_mask]
        X_te = X_imp[te_mask]
        y_te = y[te_mask]

        if len(y_te.unique()) < 2:
            print(f"    [{held_out}]: SKIP — single class")
            continue

        pos_w = (y_tr==0).sum() / max((y_tr==1).sum(), 1)

        m = XGBClassifier(
            max_depth        = 3,
            n_estimators     = 100,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 3,
            scale_pos_weight = pos_w,
            random_state     = 42,
            eval_metric      = 'auc',
            verbosity        = 0
        )
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, probs)
        aucs.append(auc)

        # Top 5 features
        top5 = pd.Series(
            m.feature_importances_, index=feats
        ).nlargest(5)
        print(f"    [{held_out}] "
              f"train={tr_mask.sum()} "
              f"test={te_mask.sum()} "
              f"AUC={auc:.3f}")
        for fn, fv in top5.items():
            print(f"      {fn:35s}: {fv:.3f}")

        RocCurveDisplay.from_predictions(
            y_te, probs, ax=axes[i],
            name=f'Hold out: {held_out}'
        )
        axes[i].set_title(f'{held_out}\nAUC={auc:.3f}')
        axes[i].plot([0,1],[0,1],'k--', alpha=0.4)

    mean_auc = np.mean(aucs) if aucs else 0
    std_auc  = np.std(aucs)  if aucs else 0
    all_results[feat_label] = mean_auc

    print(f"\n  Mean AUC ({feat_label}): "
          f"{mean_auc:.3f} ± {std_auc:.3f}")

    plt.suptitle(
        f'LOSAO — {feat_label} (canal_type excluded)',
        fontsize=10)
    plt.tight_layout()
    safe = feat_label[:15].replace(' ','_').replace('+','_')
    plt.savefig(
        os.path.join(DATA_DIR, f'losao_{safe}.png'),
        dpi=150, bbox_inches='tight')
    plt.close()

print("\n── LOSAO Summary ──")
for k, v in all_results.items():
    flag = '<-- USE THIS' if 'honest' in k.lower() else ''
    print(f"  {k:35s}: {v:.3f} {flag}")

# Pick honest feature set for final model
USE_FEATURES = FEATURES_SAT
USE_IMP      = imp_sat
print(f"\nUsing: Satellite+SM (honest) — {len(USE_FEATURES)} features")
print("Rainfall excluded from model — used only in alert tier logic")

# ── 7. Final model ─────────────────────────────────────────────
print(f"\n── Final model on {len(df)} rows ──")

imp_final = SimpleImputer(strategy='median')
X_final   = pd.DataFrame(
    imp_final.fit_transform(df[USE_FEATURES]),
    columns=USE_FEATURES
)

pos_w = len(y[y==0]) / max(len(y[y==1]), 1)
model = XGBClassifier(
    max_depth        = 3,
    n_estimators     = 100,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    scale_pos_weight = pos_w,
    random_state     = 42,
    eval_metric      = 'auc',
    verbosity        = 0
)
model.fit(X_final, y)
print("Final model trained.")

# ── 8. Score ───────────────────────────────────────────────────
df['V_score_ml']    = model.predict_proba(X_final)[:, 1]
df['V_score_final'] = (0.50 * df['V_score_physical'] +
                       0.50 * df['V_score_ml'])

# ── 9. Alert tiers ─────────────────────────────────────────────
# Rainfall used HERE as a trigger layer, not in the model
# This is the correct separation: ML scores structural vulnerability
# Rainfall triggers the alert based on that vulnerability score
def assign_tier(row):
    v   = row['V_score_final']
    r3  = row.get('rain_3d',  0) or 0
    r30 = row.get('rain_30d', 0) or 0
    sm  = row.get('sm_days_above_fc', 0) or 0
    sp  = row.get('SP_risk',  0) or 0
    qr  = row.get('Q_ratio',  0) or 0

    # RED: high vulnerability AND active trigger
    if v > 0.65 and (r3 > 50 or sm > 35 or qr > 2.0):
        return 'RED'
    # AMBER: elevated vulnerability OR chronic saturation
    elif v > 0.40 or r30 > 200 or sp > 0.70:
        return 'AMBER'
    else:
        return 'GREEN'

df['alert_tier'] = df.apply(assign_tier, axis=1)

print(f"\nAlert tier — all rows:")
print(df['alert_tier'].value_counts())
print(f"\nAlert tier — breach segments (label=1):")
print(df[df['label']==1]['alert_tier'].value_counts())
print(f"\nAlert tier — control segments (label=0):")
print(df[df['label']==0]['alert_tier'].value_counts())

# Validation: what % of breach segments are RED or AMBER?
breach_flagged = df[
    (df['label']==1) &
    (df['alert_tier'].isin(['RED','AMBER']))
].shape[0]
total_breach = df[df['label']==1].shape[0]
recall = breach_flagged / total_breach * 100
print(f"\nBreach recall: {breach_flagged}/{total_breach} "
      f"= {recall:.1f}% flagged RED or AMBER")

# ── 10. SHAP ───────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_final)

# Global bar chart
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_final,
    plot_type='bar', show=False, max_display=20)
plt.title(
    'Feature importance — AP bund vulnerability model\n'
    '(canal_type excluded — pure physical features)',
    fontsize=12)
plt.tight_layout()
plt.savefig(
    os.path.join(DATA_DIR, 'shap_summary.png'),
    dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary.png")

# Beeswarm
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_final,
    show=False, max_display=20)
plt.title(
    'SHAP beeswarm — direction and magnitude\n'
    'High LSWI / high soil moisture / low NDVI = high risk',
    fontsize=11)
plt.tight_layout()
plt.savefig(
    os.path.join(DATA_DIR, 'shap_beeswarm.png'),
    dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_beeswarm.png")

# Waterfall per study area
print("\n── Highest risk breach segments ──")
for sa in ['Budameru_Krishna',
           'Annamayya_Kadapa',
           'Rayalacheruvu_Tirupati']:
    sub = df[df['study_area'] == sa]
    if len(sub) == 0:
        continue
    idx = sub['V_score_final'].idxmax()
    row = sub.loc[idx]

    print(f"\n  {sa}")
    print(f"    Segment      : {row['segment_id']}")
    print(f"    V_score      : {row['V_score_final']:.3f}")
    print(f"    Alert tier   : {row['alert_tier']}")
    print(f"    OT/SP/SL     : {row['OT_risk']:.2f} / "
          f"{row['SP_risk']:.2f} / {row['SL_risk']:.2f}")
    print(f"    Failure mode : {row['failure_mode']}")
    print(f"    era5_sm_l2   : "
          f"{round(row.get('era5_sm_l2', 0), 4)} m³/m³")
    print(f"    sm_days_fc   : "
          f"{round(row.get('sm_days_above_fc', 0), 1)} days")
    print(f"    LSWI_anom    : "
          f"{round(row.get('LSWI_anomaly', 0), 4)}")
    print(f"    S1_SMI       : "
          f"{round(row.get('S1_SMI', 0), 3)}")
    if 'rain_1d' in row:
        print(f"    rain_1d      : {row.get('rain_1d','N/A')} mm")

    plt.figure(figsize=(11, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values        = shap_values[idx],
            base_values   = explainer.expected_value,
            data          = X_final.iloc[idx],
            feature_names = USE_FEATURES
        ),
        show=False, max_display=12
    )
    plt.title(
        f"{sa} — {row['segment_id']}\n"
        f"V_score: {row['V_score_final']:.3f} | "
        f"Tier: {row['alert_tier']} | "
        f"Mode: {row['failure_mode']}",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(DATA_DIR, f"shap_waterfall_{sa}.png"),
        dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: shap_waterfall_{sa}.png")

# ── 11. Three-event comparison table ──────────────────────────
print("\n── Three-event physical signature ──")
print(f"{'Event':30s} {'rain_1d':>8} {'sm_days':>8} "
      f"{'era5_l2':>8} {'SP_risk':>8} {'Tier':>6}")
print("-" * 75)
for sa, label in [
    ('Budameru_Krishna',       'Budameru 2024'),
    ('Annamayya_Kadapa',       'Annamayya 2021'),
    ('Rayalacheruvu_Tirupati', 'Rayalacheruvu 2025'),
]:
    sub = df[df['study_area']==sa]
    if len(sub) == 0:
        continue
    idx = sub['V_score_final'].idxmax()
    r   = sub.loc[idx]
    print(
        f"{label:30s} "
        f"{r.get('rain_1d',0):>8.1f} "
        f"{r.get('sm_days_above_fc',0):>8.1f} "
        f"{r.get('era5_sm_l2',0):>8.3f} "
        f"{r.get('SP_risk',0):>8.3f} "
        f"{r.get('alert_tier','?'):>6}"
    )

# ── 12. Export ─────────────────────────────────────────────────
out_csv = os.path.join(DATA_DIR, 'features_scored_v3.csv')
df.to_csv(out_csv, index=False)

pickle.dump(model,
    open(os.path.join(DATA_DIR, 'model.pkl'), 'wb'))
pickle.dump(imp_final,
    open(os.path.join(DATA_DIR, 'imputer.pkl'), 'wb'))
pickle.dump(USE_FEATURES,
    open(os.path.join(DATA_DIR, 'feature_names.pkl'), 'wb'))

pd.DataFrame([
    {'feature_set': k, 'mean_auc': v}
    for k, v in all_results.items()
]).to_csv(
    os.path.join(DATA_DIR, 'losao_results.csv'), index=False)

print(f"\n── Saved to {DATA_DIR} ──")
print("  features_scored_v3.csv")
print("  model.pkl")
print("  imputer.pkl")
print("  feature_names.pkl")
print("  shap_summary.png")
print("  shap_beeswarm.png")
print("  shap_waterfall_Budameru_Krishna.png")
print("  shap_waterfall_Annamayya_Kadapa.png")
print("  shap_waterfall_Rayalacheruvu_Tirupati.png")
print("  losao_results.csv")
print("\nDone.")