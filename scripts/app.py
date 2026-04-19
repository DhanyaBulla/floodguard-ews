import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, MiniMap
import streamlit as st
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodGuard EWS — AP Bund Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ── Load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    breach    = gpd.read_file(os.path.join(DATA_DIR, 'breach_scored.geojson'))
    canals    = gpd.read_file(os.path.join(DATA_DIR, 'canals_scored.geojson'))
    scored_df = pd.read_csv(os.path.join(DATA_DIR, 'features_scored_v3.csv'))
    scored_df = scored_df[scored_df['label'] == 1].copy()
    return breach, canals, scored_df

breach_gdf, canals_gdf, scored_df = load_data()

# ── Colour helpers ─────────────────────────────────────────────
TIER_COLOR = {
    'RED':         '#d73027',
    'AMBER':       '#fc8d59',
    'GREEN':       '#1a9850',
    'UNMONITORED': '#888888',
}
TIER_ICON = {'RED': '🔴', 'AMBER': '🟡', 'GREEN': '🟢'}

def tier_badge(tier):
    colors = {
        'RED':   ('🔴', '#d73027', '#fff'),
        'AMBER': ('🟡', '#fc8d59', '#333'),
        'GREEN': ('🟢', '#1a9850', '#fff'),
    }
    ico, bg, fg = colors.get(tier, ('⚪','#888','#fff'))
    return (f'<span style="background:{bg};color:{fg};'
            f'padding:2px 10px;border-radius:4px;'
            f'font-size:12px;font-weight:600">'
            f'{ico} {tier}</span>')

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 FloodGuard EWS")
    st.markdown("**AI/ML Early Warning System**")
    st.markdown("Andhra Pradesh Bund Monitoring")
    st.divider()

    st.markdown("### Model Info")
    col1, col2 = st.columns(2)
    col1.metric("AUC", "0.806")
    col2.metric("Recall", "100%")
    col1.metric("Features", "28")
    col2.metric("Events", "3")
    st.divider()

    st.markdown("### 📡 Live APWRIMS Feed")
    tank_fill = st.slider(
        "Olluru tank fill %", 0, 100, 45,
        help="Live tank water level from APWRIMS"
    )
    if tank_fill >= 85:
        st.error(f"⚠️ Tank at {tank_fill}% — Rayalacheruvu segments auto-flagged AMBER")
    elif tank_fill >= 70:
        st.warning(f"Tank at {tank_fill}% — Monitor closely")
    else:
        st.success(f"Tank at {tank_fill}% — Normal")

    st.divider()
    st.markdown("### Alert Summary")
    tier_counts = scored_df['alert_tier'].value_counts()
    for tier in ['RED','AMBER','GREEN']:
        count = tier_counts.get(tier, 0)
        st.markdown(
            f"{TIER_ICON.get(tier,'⚪')} **{tier}**: {count} segments")

    st.divider()
    st.markdown("### Data Sources")
    st.markdown("""
    - Sentinel-2 (NDVI, LSWI, NDWI)
    - Sentinel-1 SAR (VV, VH, SMI)
    - SMAP soil moisture
    - ERA5-Land soil water
    - IMD gridded rainfall
    - APWRIMS tank levels
    - SRTM terrain (slope, TWI)
    """)
    # ── Main tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🗺️ Risk Map",
    "📊 Risk Table",
    "🔍 Segment Detail"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### AP Bund Vulnerability Map")

    col1, col2, col3, col4 = st.columns(4)
    total       = len(scored_df)
    red_count   = (scored_df['alert_tier'] == 'RED').sum()
    amber_count = (scored_df['alert_tier'] == 'AMBER').sum()
    green_count = (scored_df['alert_tier'] == 'GREEN').sum()

    col1.metric("Total segments", total)
    col2.metric("🔴 Red (inspect)", red_count)
    col3.metric("🟡 Amber (monitor)", amber_count)
    col4.metric("🟢 Green (routine)", green_count)

    st.divider()

    # Map filters
    fc1, fc2, fc3 = st.columns(3)
    filter_tier = fc1.multiselect(
        "Filter by tier",
        ['RED','AMBER','GREEN'],
        default=['RED','AMBER','GREEN']
    )
    filter_area = fc2.multiselect(
        "Filter by study area",
        scored_df['study_area'].unique().tolist(),
        default=scored_df['study_area'].unique().tolist()
    )
    show_canals = fc3.checkbox("Show canal infrastructure", value=True)

    # Filter scored_df
    map_df = scored_df[
        (scored_df['alert_tier'].isin(filter_tier)) &
        (scored_df['study_area'].isin(filter_area))
    ].copy()

    # Apply APWRIMS boost
    if tank_fill >= 85:
        raya_idx = map_df[
            map_df['study_area'] == 'Rayalacheruvu_Tirupati'
        ].index
        map_df.loc[raya_idx, 'alert_tier'] = 'AMBER'

    # Build Folium map
    m = folium.Map(
        location=[15.9, 80.0],
        zoom_start=7,
        tiles='CartoDB positron'
    )

    # Canal infrastructure layer (gray lines)
    if show_canals:
        canal_layer = folium.FeatureGroup(
            name="Canal infrastructure (unmonitored)", show=True)
        for _, row in canals_gdf.iterrows():
            try:
                if row.geometry is None:
                    continue
                geom_type = row.geometry.geom_type
                if geom_type == 'LineString':
                    coords = [(y, x) for x, y in
                              row.geometry.coords]
                elif geom_type == 'MultiLineString':
                    coords = [(y, x) for x, y in
                              row.geometry.geoms[0].coords]
                else:
                    continue

                folium.PolyLine(
                    coords,
                    color='#888888',
                    weight=1,
                    opacity=0.4,
                    tooltip=folium.Tooltip(
                        f"{row.get('parent_name','Canal')} "
                        f"({row.get('canal_type','')}) — Unmonitored"
                    )
                ).add_to(canal_layer)
            except Exception:
                continue
        canal_layer.add_to(m)

    # Breach segments — colored circles
    breach_layer = folium.FeatureGroup(
        name="Scored breach segments", show=True)

    for _, row in map_df.iterrows():
        tier  = row.get('alert_tier', 'GREEN')
        color = TIER_COLOR.get(tier, '#888888')
        lat   = row.get('centroid_lat') or row.get('lat')
        lon   = row.get('centroid_lon') or row.get('lon')
        if lat is None or lon is None:
            continue

        # Dominant failure mode
        scores = {
            'Overtopping':       row.get('OT_risk', 0),
            'Seepage/Piping':    row.get('SP_risk', 0),
            'Slope Instability': row.get('SL_risk', 0),
        }
        dom_mode = max(scores, key=scores.get)

        popup_html = f"""
        <div style="font-family:Arial;width:260px">
          <h4 style="margin:0;color:{color}">
            {TIER_ICON.get(tier,'⚪')} {tier} ALERT
          </h4>
          <hr style="margin:4px 0">
          <b>Segment:</b> {row.get('segment_id','')}<br>
          <b>Study area:</b> {row.get('study_area','')}<br>
          <b>District:</b> {row.get('district','')}<br>
          <b>Failure mode:</b> {row.get('failure_mode','')}<br>
          <hr style="margin:4px 0">
          <b>V-score:</b> {row.get('V_score_final',0):.3f}<br>
          <b>Dominant mechanism:</b> {dom_mode}<br>
          <hr style="margin:4px 0">
          <b>OT risk:</b> {row.get('OT_risk',0):.2f} &nbsp;
          <b>SP risk:</b> {row.get('SP_risk',0):.2f} &nbsp;
          <b>SL risk:</b> {row.get('SL_risk',0):.2f}<br>
          <hr style="margin:4px 0">
          <b>ERA5 soil moisture (7-28cm):</b>
            {row.get('era5_sm_l2',0):.3f} m³/m³<br>
          <b>Days above field capacity:</b>
            {row.get('sm_days_above_fc',0):.0f} days<br>
          <b>LSWI anomaly:</b>
            {row.get('LSWI_anomaly',0):.3f}<br>
          <b>Rain 1-day:</b>
            {row.get('rain_1d',0):.1f} mm<br>
          <b>Rain 30-day:</b>
            {row.get('rain_30d',0):.1f} mm<br>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=folium.Tooltip(
                f"{tier} | V={row.get('V_score_final',0):.2f} "
                f"| {dom_mode}"
            )
        ).add_to(breach_layer)

    breach_layer.add_to(m)

    # Study area labels
    study_centers = {
        'Budameru_Krishna':       (16.52, 80.60, 'Budameru 2024'),
        'Annamayya_Kadapa':       (14.18, 79.15, 'Annamayya 2021'),
        'Rayalacheruvu_Tirupati': (13.45, 79.89, 'Rayalacheruvu 2025'),
    }
    for sa, (lat, lon, label) in study_centers.items():
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;'
                     f'font-weight:bold;color:#333;'
                     f'background:rgba(255,255,255,0.8);'
                     f'padding:2px 6px;border-radius:3px;'
                     f'white-space:nowrap">{label}</div>',
                icon_size=(120, 20),
                icon_anchor=(60, 10)
            )
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;
                background:white;padding:12px 16px;
                border-radius:8px;border:1px solid #ccc;
                font-family:Arial;font-size:13px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.15);
                z-index:1000">
      <b>Vulnerability Tier</b><br>
      <span style="color:#d73027">●</span>
        RED — Inspect within 24h<br>
      <span style="color:#fc8d59">●</span>
        AMBER — Monitor daily<br>
      <span style="color:#1a9850">●</span>
        GREEN — Routine inspection<br>
      <span style="color:#888888">—</span>
        Canal infrastructure
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Minimap
    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)

    st_folium(m, width=None, height=580, returned_objects=[])

# ══════════════════════════════════════════════════════════════
# TAB 2 — RISK TABLE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Segment Risk Table")

    # Filters
    tc1, tc2, tc3 = st.columns(3)
    t_area = tc1.multiselect(
        "Study area",
        scored_df['study_area'].unique().tolist(),
        default=scored_df['study_area'].unique().tolist()
    )
    t_tier = tc2.multiselect(
        "Alert tier",
        ['RED','AMBER','GREEN'],
        default=['RED','AMBER','GREEN']
    )
    t_sort = tc3.selectbox(
        "Sort by",
        ['V_score_final','OT_risk','SP_risk','SL_risk'],
        index=0
    )

    table_df = scored_df[
        (scored_df['study_area'].isin(t_area)) &
        (scored_df['alert_tier'].isin(t_tier))
    ].sort_values(t_sort, ascending=False).copy()

    # Display columns
    display_cols = [
        'segment_id','study_area','district',
        'alert_tier','V_score_final',
        'OT_risk','SP_risk','SL_risk',
        'failure_mode','era5_sm_l2',
        'sm_days_above_fc'
    ]
    display_cols = [c for c in display_cols
                    if c in table_df.columns]

    st.markdown(f"**{len(table_df)} segments shown**")

    # Color the alert_tier column
    def color_tier(val):
        colors = {
            'RED':   'background-color:#d73027;color:white',
            'AMBER': 'background-color:#fc8d59;color:#333',
            'GREEN': 'background-color:#1a9850;color:white',
        }
        return colors.get(val, '')

    styled = (table_df[display_cols]
              .reset_index(drop=True)
              .style
              .applymap(color_tier, subset=['alert_tier'])
              .format({
                  'V_score_final':     '{:.3f}',
                  'OT_risk':           '{:.3f}',
                  'SP_risk':           '{:.3f}',
                  'SL_risk':           '{:.3f}',
                  'era5_sm_l2':        '{:.3f}',
                  'sm_days_above_fc':  '{:.0f}',
              }))

    st.dataframe(styled, use_container_width=True, height=500)

    # Download
    csv_data = table_df[display_cols].to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="bund_risk_report.csv",
        mime="text/csv"
    )

    st.divider()

    # Three-event comparison bar chart
    st.markdown("### Three-event physical signature")
    st.caption(
        "Each event has a distinct rainfall and soil moisture profile — "
        "demonstrating why multi-source monitoring is essential"
    )

    event_data = []
    for sa in ['Budameru_Krishna','Annamayya_Kadapa',
               'Rayalacheruvu_Tirupati']:
        sub = scored_df[scored_df['study_area'] == sa]
        if len(sub) == 0:
            continue
        worst = sub.loc[sub['V_score_final'].idxmax()]
        event_data.append({
            'Event':         sa.replace('_',' '),
            'rain_1d (mm)':  round(worst.get('rain_1d', 0), 1),
            'sm_days_fc':    round(worst.get('sm_days_above_fc', 0), 0),
            'era5_sm_l2':    round(worst.get('era5_sm_l2', 0), 3),
            'SP_risk':       round(worst.get('SP_risk', 0), 3),
            'V_score':       round(worst.get('V_score_final', 0), 3),
            'Tier':          worst.get('alert_tier', ''),
        })

    if event_data:
        edf = pd.DataFrame(event_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='1-day rainfall (mm)',
            x=edf['Event'],
            y=edf['rain_1d (mm)'],
            marker_color='#4393c3',
            yaxis='y'
        ))
        fig.add_trace(go.Bar(
            name='Days above field capacity',
            x=edf['Event'],
            y=edf['sm_days_fc'],
            marker_color='#d6604d',
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            name='V_score (right)',
            x=edf['Event'],
            y=edf['V_score'],
            mode='lines+markers',
            marker=dict(size=10, color='#333'),
            line=dict(width=2, dash='dot'),
            yaxis='y2'
        ))
        fig.update_layout(
            barmode='group',
            height=350,
            yaxis=dict(title='mm / days'),
            yaxis2=dict(
                title='V_score',
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            legend=dict(
                orientation='h', y=-0.2),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — SEGMENT DETAIL
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Segment Detail — Vulnerability Explainer")

    dc1, dc2 = st.columns([1, 2])

    with dc1:
        selected = st.selectbox(
            "Select segment",
            scored_df.sort_values(
                'V_score_final', ascending=False
            )['segment_id'].tolist(),
            key='seg_detail'
        )

        seg = scored_df[
            scored_df['segment_id'] == selected].iloc[0]

        tier  = seg.get('alert_tier', 'GREEN')
        color = TIER_COLOR.get(tier, '#888')

        st.markdown(
            f"**Alert tier:** {tier_badge(tier)}",
            unsafe_allow_html=True
        )
        st.metric("V-score (final)",
                  f"{seg.get('V_score_final',0):.3f}")

        st.divider()
        st.markdown("**Key physical indicators**")

        indicators = {
            'ERA5 soil moisture (7-28cm)':
                f"{seg.get('era5_sm_l2',0):.3f} m³/m³",
            'Days above field capacity':
                f"{seg.get('sm_days_above_fc',0):.0f} days",
            'SMAP soil moisture':
                f"{seg.get('smap_sm_mean',0):.3f} m³/m³",
            'LSWI anomaly':
                f"{seg.get('LSWI_anomaly',0):.3f}",
            'S1 soil moisture index':
                f"{seg.get('S1_SMI',0):.3f}",
            'Slope (°)':
                f"{seg.get('slope',0):.2f}",
            'TWI':
                f"{seg.get('TWI',0):.3f}",
            'Encroachment %':
                f"{seg.get('encroachment_pct',0)*100:.1f}%",
            'Rain 1-day':
                f"{seg.get('rain_1d',0):.1f} mm",
            'Rain 30-day':
                f"{seg.get('rain_30d',0):.1f} mm",
        }
        for k, v in indicators.items():
            st.markdown(
                f"<div style='display:flex;"
                f"justify-content:space-between;"
                f"font-size:13px;padding:2px 0'>"
                f"<span style='color:#666'>{k}</span>"
                f"<span style='font-weight:500'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    with dc2:
        # Radar chart — OT/SP/SL sub-scores
        ot = float(seg.get('OT_risk', 0))
        sp = float(seg.get('SP_risk', 0))
        sl = float(seg.get('SL_risk', 0))

        categories = [
            'Overtopping\nRisk',
            'Seepage/\nPiping Risk',
            'Slope\nInstability',
        ]
        values = [ot, sp, sl]
        values_closed = values + [values[0]]
        cats_closed   = categories + [categories[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill='toself',
            fillcolor=f'rgba({int(color[1:3],16)},'
                      f'{int(color[3:5],16)},'
                      f'{int(color[5:7],16)},0.25)',
            line=dict(color=color, width=2),
            name='Risk sub-scores'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    tickfont=dict(size=10)
                )
            ),
            showlegend=False,
            height=320,
            margin=dict(t=20, b=20, l=40, r=40),
            title=dict(
                text=f"Failure mode breakdown — {selected}",
                font=dict(size=13)
            )
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Physical vs ML score comparison
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='Physical score',
            x=['Physical\n(OT+SP+SL)','ML score','Final\n(blend)'],
            y=[
                round(float(seg.get('V_score_physical',0)),3),
                round(float(seg.get('V_score_ml',0)),3),
                round(float(seg.get('V_score_final',0)),3),
            ],
            marker_color=[color, '#4393c3', color],
            text=[
                f"{seg.get('V_score_physical',0):.3f}",
                f"{seg.get('V_score_ml',0):.3f}",
                f"{seg.get('V_score_final',0):.3f}",
            ],
            textposition='outside'
        ))
        fig_bar.update_layout(
            height=250,
            yaxis=dict(range=[0, 1.1], title='Score'),
            showlegend=False,
            margin=dict(t=20, b=20),
            title=dict(
                text="Score decomposition",
                font=dict(size=13)
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # SHAP waterfall image
    st.divider()
    st.markdown("### SHAP explanation — why this segment scores high")

    study_area = seg.get('study_area', '')
    shap_file  = os.path.join(
        DATA_DIR, f'shap_waterfall_{study_area}.png')

    if os.path.exists(shap_file):
        img = Image.open(shap_file)
        st.image(img, use_column_width=True,
                 caption=f"SHAP waterfall — {study_area}")
        st.caption(
            "Each bar shows how much a feature pushes the "
            "vulnerability score up (red) or down (blue) "
            "from the baseline. Features are ordered by impact."
        )
    else:
        st.info(
            f"SHAP plot not found for {study_area}. "
            f"Run enrich_and_train.py to generate."
        )

    # Study area context
    st.divider()
    st.markdown("### Study area context")

    context = {
        'Budameru_Krishna': {
            'title': 'Budameru Diversion Canal — Vijayawada, Aug 31 2024',
            'desc':  ('Canal flow hit 990 m³/s against 200 m³/s design capacity. '
                      'Bund breached, 270,000 people affected, 35 deaths. '
                      'Bund body was saturated for 38/62 days pre-breach. '
                      'Urban encroachment (60% of buffer) restricted maintenance access.'),
            'mode':  'Overtopping + saturated bund body',
        },
        'Annamayya_Kadapa': {
            'title': 'Annamayya Project — Cheyyeru river, Kadapa, Nov 19 2021',
            'desc':  ('Upstream Pincha project breached → 2 lakh cusecs into Annamayya. '
                      'Earthen bund body saturated for 99% of 49-day pre-breach window. '
                      '18 deaths, 10 villages inundated. Red laterite soil — faster piping.'),
            'mode':  'Cascade failure + piping',
        },
        'Rayalacheruvu_Tirupati': {
            'title': 'Olluru Rayalacheruvu Tank — KVB Puram, Tirupati, Nov 6 2025',
            'desc':  ('Tank filled to 100% over weeks of sustained rain. '
                      'Bund breached on a day with only 0.23mm rainfall. '
                      'Farmers had reported bund deterioration — no action taken. '
                      '5 villages inundated. Detected by soil moisture, not rainfall.'),
            'mode':  'Chronic saturation — dry day breach',
        },
    }

    if study_area in context:
        ctx = context[study_area]
        st.markdown(f"**{ctx['title']}**")
        st.markdown(ctx['desc'])
        st.markdown(
            f"**Primary failure mode:** `{ctx['mode']}`")

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px'>"
    "FloodGuard EWS · Built for APSDMA/RTGS AI Hackathon · "
    "Data: Sentinel-1/2, SMAP, ERA5-Land, IMD, APWRIMS"
    "</div>",
    unsafe_allow_html=True
)