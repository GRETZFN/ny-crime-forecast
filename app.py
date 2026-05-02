from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NY County Crime Forecast Explorer",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent / "data"

MODEL_COLORS = {
    "Linear Regression": "#1f77b4",
    "Random Forest": "#2ca02c",
    "Gradient Boosting (tuned)": "#ff7f0e",
}

SERIES_COLORS = {
    "Actual":      "#222222",
    "Actual_2yr":  "#666666",
    "GB_Pred_1yr": "#86b5ff",
    "GB_Pred_2yr": "#ff7f0e",
}

SERIES_LABELS = {
    "Actual":      "Actual (1-yr folds)",
    "Actual_2yr":  "Actual (2-yr folds)",
    "GB_Pred_1yr": "GB Predicted (1-Year)",
    "GB_Pred_2yr": "GB Predicted (2-Year)",
}

# 5-digit FIPS codes for all 62 NY counties
NY_COUNTY_FIPS: dict[str, str] = {
    "Albany": "36001",       "Allegany": "36003",     "Bronx": "36005",
    "Broome": "36007",       "Cattaraugus": "36009",  "Cayuga": "36011",
    "Chautauqua": "36013",   "Chemung": "36015",      "Chenango": "36017",
    "Clinton": "36019",      "Columbia": "36021",     "Cortland": "36023",
    "Delaware": "36025",     "Dutchess": "36027",     "Erie": "36029",
    "Essex": "36031",        "Franklin": "36033",     "Fulton": "36035",
    "Genesee": "36037",      "Greene": "36039",       "Hamilton": "36041",
    "Herkimer": "36043",     "Jefferson": "36045",    "Kings": "36047",
    "Lewis": "36049",        "Livingston": "36051",   "Madison": "36053",
    "Monroe": "36055",       "Montgomery": "36057",   "Nassau": "36059",
    "New York": "36061",     "Niagara": "36063",      "Oneida": "36065",
    "Onondaga": "36067",     "Ontario": "36069",      "Orange": "36071",
    "Orleans": "36073",      "Oswego": "36075",       "Otsego": "36077",
    "Putnam": "36079",       "Queens": "36081",       "Rensselaer": "36083",
    "Richmond": "36085",     "Rockland": "36087",     "St. Lawrence": "36089",
    "Saratoga": "36091",     "Schenectady": "36093",  "Schoharie": "36095",
    "Schuyler": "36097",     "Seneca": "36099",       "Steuben": "36101",
    "Suffolk": "36103",      "Sullivan": "36105",     "Tioga": "36107",
    "Tompkins": "36109",     "Ulster": "36111",       "Warren": "36113",
    "Washington": "36115",   "Wayne": "36117",        "Westchester": "36119",
    "Wyoming": "36121",      "Yates": "36123",
}

# Map any alternative county names in the data to the GeoJSON canonical name
COUNTY_NAME_MAP: dict[str, str] = {
    "Brooklyn":      "Kings",
    "Manhattan":     "New York",
    "Staten Island": "Richmond",
    "Saint Lawrence": "St. Lawrence",
}

# NYS standard 10-region grouping
NY_COUNTY_REGION: dict[str, str] = {
    "Bronx":        "New York City", "Kings":        "New York City",
    "New York":     "New York City", "Queens":       "New York City",
    "Richmond":     "New York City",
    "Nassau":       "Long Island",   "Suffolk":      "Long Island",
    "Dutchess":     "Hudson Valley", "Orange":       "Hudson Valley",
    "Putnam":       "Hudson Valley", "Rockland":     "Hudson Valley",
    "Sullivan":     "Hudson Valley", "Ulster":       "Hudson Valley",
    "Westchester":  "Hudson Valley",
    "Albany":       "Capital Region","Columbia":     "Capital Region",
    "Greene":       "Capital Region","Rensselaer":   "Capital Region",
    "Saratoga":     "Capital Region","Schenectady":  "Capital Region",
    "Schoharie":    "Capital Region","Warren":       "Capital Region",
    "Washington":   "Capital Region",
    "Fulton":       "Mohawk Valley", "Hamilton":     "Mohawk Valley",
    "Herkimer":     "Mohawk Valley", "Montgomery":   "Mohawk Valley",
    "Oneida":       "Mohawk Valley", "Otsego":       "Mohawk Valley",
    "Clinton":      "North Country", "Essex":        "North Country",
    "Franklin":     "North Country", "Jefferson":    "North Country",
    "Lewis":        "North Country", "St. Lawrence": "North Country",
    "Cayuga":       "Central NY",    "Cortland":     "Central NY",
    "Madison":      "Central NY",    "Onondaga":     "Central NY",
    "Oswego":       "Central NY",
    "Broome":       "Southern Tier", "Chemung":      "Southern Tier",
    "Chenango":     "Southern Tier", "Delaware":     "Southern Tier",
    "Schuyler":     "Southern Tier", "Steuben":      "Southern Tier",
    "Tioga":        "Southern Tier", "Tompkins":     "Southern Tier",
    "Genesee":      "Finger Lakes",  "Livingston":   "Finger Lakes",
    "Monroe":       "Finger Lakes",  "Ontario":      "Finger Lakes",
    "Orleans":      "Finger Lakes",  "Seneca":       "Finger Lakes",
    "Wayne":        "Finger Lakes",  "Wyoming":      "Finger Lakes",
    "Yates":        "Finger Lakes",
    "Allegany":     "Western NY",    "Cattaraugus":  "Western NY",
    "Chautauqua":   "Western NY",    "Erie":         "Western NY",
    "Niagara":      "Western NY",
}

# Approximate centroids (lat, lon) for all 62 NY counties
NY_COUNTY_CENTROIDS: dict[str, tuple[float, float]] = {
    "Albany":       (42.60, -73.97),  "Allegany":     (42.25, -78.03),
    "Bronx":        (40.84, -73.86),  "Broome":       (42.17, -75.83),
    "Cattaraugus":  (42.25, -78.67),  "Cayuga":       (42.95, -76.57),
    "Chautauqua":   (42.30, -79.38),  "Chemung":      (42.15, -76.77),
    "Chenango":     (42.50, -75.62),  "Clinton":      (44.77, -73.68),
    "Columbia":     (42.25, -73.63),  "Cortland":     (42.60, -76.08),
    "Delaware":     (42.20, -74.97),  "Dutchess":     (41.77, -73.75),
    "Erie":         (42.75, -78.78),  "Essex":        (44.12, -73.77),
    "Franklin":     (44.60, -74.30),  "Fulton":       (43.13, -74.42),
    "Genesee":      (43.00, -78.18),  "Greene":       (42.27, -74.25),
    "Hamilton":     (43.70, -74.52),  "Herkimer":     (43.43, -74.97),
    "Jefferson":    (43.98, -75.98),  "Kings":        (40.64, -73.95),
    "Lewis":        (43.78, -75.45),  "Livingston":   (42.72, -77.78),
    "Madison":      (42.93, -75.67),  "Monroe":       (43.17, -77.62),
    "Montgomery":   (42.90, -74.43),  "Nassau":       (40.73, -73.59),
    "New York":     (40.78, -73.97),  "Niagara":      (43.22, -78.95),
    "Oneida":       (43.25, -75.45),  "Onondaga":     (43.00, -76.18),
    "Ontario":      (42.85, -77.30),  "Orange":       (41.38, -74.30),
    "Orleans":      (43.25, -78.18),  "Oswego":       (43.47, -76.22),
    "Otsego":       (42.63, -75.00),  "Putnam":       (41.43, -73.77),
    "Queens":       (40.71, -73.79),  "Rensselaer":   (42.72, -73.50),
    "Richmond":     (40.58, -74.15),  "Rockland":     (41.15, -74.05),
    "St. Lawrence": (44.50, -75.00),  "Saratoga":     (43.10, -73.87),
    "Schenectady":  (42.82, -74.08),  "Schoharie":    (42.58, -74.45),
    "Schuyler":     (42.40, -76.87),  "Seneca":       (42.78, -76.83),
    "Steuben":      (42.27, -77.38),  "Suffolk":      (40.92, -72.68),
    "Sullivan":     (41.73, -74.78),  "Tioga":        (42.17, -76.30),
    "Tompkins":     (42.45, -76.48),  "Ulster":       (41.87, -74.27),
    "Warren":       (43.65, -73.85),  "Washington":   (43.32, -73.43),
    "Wayne":        (43.08, -76.98),  "Westchester":  (41.13, -73.78),
    "Wyoming":      (42.70, -78.22),  "Yates":        (42.63, -77.12),
}


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "annual": pd.read_csv(DATA_DIR / "annual_county_rate_interp.csv"),
        "summary_1yr": pd.read_csv(DATA_DIR / "backtest_1yr_summary.csv"),
        "summary_2yr": pd.read_csv(DATA_DIR / "backtest_2yr_summary.csv"),
        "results_1yr": pd.read_csv(DATA_DIR / "backtest_1yr_results.csv"),
        "results_2yr": pd.read_csv(DATA_DIR / "backtest_2yr_results.csv"),
        "preds_1yr": pd.read_csv(DATA_DIR / "backtest_1yr_predictions.csv"),
        "preds_2yr": pd.read_csv(DATA_DIR / "backtest_2yr_predictions.csv"),
        "config": pd.read_csv(DATA_DIR / "model_config.csv"),
    }
    for key in ["annual", "preds_1yr", "preds_2yr", "results_1yr", "results_2yr"]:
        if "Year" in data[key].columns:
            data[key]["Year"] = pd.to_numeric(data[key]["Year"], errors="coerce")
        if "target_year" in data[key].columns:
            data[key]["target_year"] = pd.to_numeric(data[key]["target_year"], errors="coerce")
        if "FoldYear" in data[key].columns:
            data[key]["FoldYear"] = pd.to_numeric(data[key]["FoldYear"], errors="coerce")
    return data


def model_palette(models: list[str]) -> dict[str, str]:
    palette = MODEL_COLORS.copy()
    extra = [m for m in models if not any(k in m for k in MODEL_COLORS)]
    fallbacks = ["#d62728", "#9467bd", "#8c564b"]
    for i, m in enumerate(extra):
        palette[m] = fallbacks[i % len(fallbacks)]
    return palette


def build_horizon_compare(summary_1yr: pd.DataFrame, summary_2yr: pd.DataFrame) -> pd.DataFrame:
    joined = summary_1yr.merge(summary_2yr, on="Model", suffixes=("_1yr", "_2yr"))
    joined["MAE_Gap"] = joined["MAE_2yr"] - joined["MAE_1yr"]
    return joined.sort_values("MAE_2yr").reset_index(drop=True)


def build_county_snapshot(preds_1yr: pd.DataFrame, preds_2yr: pd.DataFrame, county: str) -> pd.DataFrame:
    one = (
        preds_1yr[preds_1yr["County"] == county][["County", "target_year", "Actual_tplus1", "Pred_GB_Tuned"]]
        .copy()
        .rename(columns={"Actual_tplus1": "Actual", "Pred_GB_Tuned": "GB_Pred_1yr"})
    )
    two = (
        preds_2yr[preds_2yr["County"] == county][["County", "target_year", "Actual_tplus2", "Pred_GB_Tuned"]]
        .copy()
        .rename(columns={"Actual_tplus2": "Actual_2yr", "Pred_GB_Tuned": "GB_Pred_2yr"})
    )
    return one.merge(two, on=["County", "target_year"], how="outer").sort_values("target_year")


@st.cache_data(show_spinner="Loading county boundaries…")
def load_ny_geojson() -> dict | None:
    import json
    import urllib.request
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            all_counties = json.loads(response.read())
        return {
            "type": "FeatureCollection",
            "features": [f for f in all_counties["features"] if f["id"].startswith("36")],
        }
    except Exception:
        return None


def _geojson_centroid(feature: dict) -> tuple[float, float] | None:
    """Return approximate (lat, lon) centroid from a GeoJSON polygon/multipolygon."""
    flat: list[list[float]] = []

    def _collect(obj: list) -> None:
        if obj and isinstance(obj[0], list):
            for child in obj:
                _collect(child)
        else:
            flat.append(obj)

    _collect(feature["geometry"]["coordinates"])
    if not flat:
        return None
    lon = sum(c[0] for c in flat) / len(flat)
    lat = sum(c[1] for c in flat) / len(flat)
    return lat, lon


def with_fold_year(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "FoldYear" in normalized.columns:
        normalized["FoldYear"] = pd.to_numeric(normalized["FoldYear"], errors="coerce")
        return normalized

    if "Fold" in normalized.columns:
        normalized["FoldYear"] = (
            normalized["Fold"]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
        )
        normalized["FoldYear"] = pd.to_numeric(normalized["FoldYear"], errors="coerce")
        return normalized

    normalized["FoldYear"] = pd.NA
    return normalized


data = load_data()
summary_1yr = data["summary_1yr"].copy()
summary_2yr = data["summary_2yr"].copy()
results_1yr = with_fold_year(data["results_1yr"])
results_2yr = with_fold_year(data["results_2yr"])
preds_1yr = data["preds_1yr"].copy()
preds_2yr = data["preds_2yr"].copy()
annual = data["annual"].copy()
config = data["config"].copy()

horizon_compare = build_horizon_compare(summary_1yr, summary_2yr)
models = horizon_compare["Model"].tolist()
palette = model_palette(models)

best_1yr = summary_1yr.sort_values("MAE").iloc[0]
best_2yr = summary_2yr.sort_values("MAE").iloc[0]
counties = sorted(annual["County"].dropna().unique())
default_county = "Albany" if "Albany" in counties else counties[0]

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🗺 NY Crime Explorer")
    st.markdown("Capstone project — NY county crime rate forecasting")
    st.divider()

    st.subheader("Filters")
    selected_county = st.selectbox("County", counties, index=counties.index(default_county))
    selected_model  = st.selectbox(
        "Model (fold diagnostics)",
        models,
        index=models.index("Gradient Boosting (tuned)") if "Gradient Boosting (tuned)" in models else 0,
    )

    st.divider()
    st.subheader("Recommended model")
    st.success(best_2yr["Model"])
    if not config.empty:
        gb_params = config.loc[config["key"] == "best_tuned_gb_params", "value"]
        meta_name = config.loc[config["key"] == "best_tuned_stack_meta", "value"]
        if not gb_params.empty:
            with st.expander("GB hyperparameters"):
                st.code(gb_params.iloc[0])
        if not meta_name.empty:
            with st.expander("Stack meta-learner"):
                st.code(meta_name.iloc[0])

    st.divider()
    st.caption("Data: NYS Index Crimes by County & Agency (1990–2024)")

# ── page header ───────────────────────────────────────────────────────────────

st.title("New York County Crime Rate Forecast")
st.markdown(
    "This dashboard presents the results of an expanding-window backtest comparing "
    "four models at **1-year** and **2-year** forecast horizons across all NY counties. "
    "Use the sidebar to explore results by county and model."
)

# ── KPI row ──────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best 1-Year MAE",          f"{best_1yr['MAE']:.1f}",                           best_1yr["Model"])
k2.metric("Best 2-Year MAE",          f"{best_2yr['MAE']:.1f}",                           best_2yr["Model"])
k3.metric("Horizon Accuracy Penalty", f"+{best_2yr['MAE'] - best_1yr['MAE']:.1f}",        "2yr vs 1yr (best model)")
k4.metric("Counties Covered",         f"{annual['County'].nunique()}")

st.info(
    "The 1-year horizon achieves lower MAE for every model because it is an inherently easier forecasting task. "
    "The **2-year Gradient Boosting model** is recommended for deployment because it aligns with planning lead-time "
    "requirements while still being the strongest model at the harder horizon."
)

st.divider()

# ── county crime map ──────────────────────────────────────────────────────────

st.subheader("County crime rate map")
_geojson = load_ny_geojson()
if _geojson is None:
    st.warning("County boundary data could not be loaded. Check your internet connection.")
else:
    _year_min = int(annual["Year"].dropna().min())
    _year_max = int(annual["Year"].dropna().max())
    map_year = st.slider(
        "Select year", min_value=_year_min, max_value=_year_max,
        value=_year_max, step=1, format="%d",
    )

    map_df = annual[annual["Year"] == map_year].copy()
    map_df["GeoName"] = map_df["County"].replace(COUNTY_NAME_MAP)
    map_df["fips"]    = map_df["GeoName"].map(NY_COUNTY_FIPS)
    map_df = map_df.dropna(subset=["fips", "Crime_Rate_per_100k"])

    fig_map = px.choropleth_mapbox(
        map_df,
        geojson=_geojson,
        locations="fips",
        color="Crime_Rate_per_100k",
        color_continuous_scale="Reds",
        mapbox_style="open-street-map",
        zoom=5.5,
        center={"lat": 42.9, "lon": -76.0},
        opacity=0.7,
        hover_name="County",
        hover_data={"fips": False, "Crime_Rate_per_100k": ":.1f"},
        labels={"Crime_Rate_per_100k": "Crime rate / 100k"},
        title=f"Crime rate per 100k — {map_year}",
    )

    # Gold pin on the selected county
    _geo_name = COUNTY_NAME_MAP.get(selected_county, selected_county)
    _sel_fips = NY_COUNTY_FIPS.get(_geo_name)
    if _sel_fips:
        _sel_feat = next((f for f in _geojson["features"] if f["id"] == _sel_fips), None)
        if _sel_feat:
            _centroid = _geojson_centroid(_sel_feat)
            if _centroid:
                fig_map.add_trace(go.Scattermapbox(
                    lat=[_centroid[0]], lon=[_centroid[1]],
                    mode="markers+text",
                    marker=dict(size=14, color="gold"),
                    text=[selected_county],
                    textposition="top right",
                    name=selected_county,
                    hoverinfo="name",
                ))

    fig_map.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=500,
        coloraxis_colorbar=dict(title="per 100k"),
    )
    st.plotly_chart(fig_map, width="stretch")

st.divider()

# ── horizon comparison ───────────────────────────────────────────────────────

st.subheader("Model comparison across horizons")
left, right = st.columns([1.1, 0.9])

with left:
    long = horizon_compare.melt(
        id_vars=["Model", "MAE_Gap"],
        value_vars=["MAE_1yr", "MAE_2yr"],
        var_name="Horizon",
        value_name="MAE",
    )
    long["Horizon"] = long["Horizon"].map({"MAE_1yr": "1-Year Ahead", "MAE_2yr": "2-Year Ahead"})
    long["Short"]   = (long["Model"]
                       .str.replace(" Regression", " Reg.", regex=False)
                       .str.replace(" (tuned)", "", regex=False)
                       .str.replace("Stacking tuned ", "Stack ", regex=False))
    fig_bar = px.bar(
        long, x="Short", y="MAE", color="Horizon", barmode="group",
        color_discrete_sequence=["#86b5ff", "#264653"],
        labels={"Short": "", "MAE": "MAE (crimes per 100k)"},
        title="Aggregate MAE — All Models",
    )
    fig_bar.update_layout(legend_title=None, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_bar, width="stretch")

with right:
    gap = horizon_compare[["Model", "MAE_Gap"]].sort_values("MAE_Gap")
    gap = gap.copy()
    gap["Short"] = (gap["Model"]
                    .str.replace(" Regression", " Reg.", regex=False)
                    .str.replace(" (tuned)", "", regex=False)
                    .str.replace("Stacking tuned ", "Stack ", regex=False))
    fig_gap = px.bar(
        gap, x="MAE_Gap", y="Short", orientation="h",
        color="Model", color_discrete_map=palette,
        labels={"MAE_Gap": "Extra MAE (2yr − 1yr)", "Short": ""},
        title="Horizon Accuracy Penalty per Model",
    )
    fig_gap.update_layout(showlegend=False)
    st.plotly_chart(fig_gap, width="stretch")

st.divider()

# ── fold diagnostics + county history ────────────────────────────────────────

st.subheader(f"Fold-by-fold diagnostics — {selected_model}")
fold_left, fold_right = st.columns(2)

with fold_left:
    m1 = results_1yr[results_1yr["Model"] == selected_model].sort_values("FoldYear")
    m2 = results_2yr[results_2yr["Model"] == selected_model].sort_values("FoldYear")
    fig_fold = go.Figure()
    fig_fold.add_trace(go.Scatter(
        x=m1["FoldYear"], y=m1["MAE"], mode="lines+markers",
        name="1-Year Ahead",
        line=dict(color="#86b5ff", width=2.5), marker=dict(size=7),
    ))
    fig_fold.add_trace(go.Scatter(
        x=m2["FoldYear"], y=m2["MAE"], mode="lines+markers",
        name="2-Year Ahead",
        line=dict(color=palette.get(selected_model, "#ff7f0e"), width=2.5), marker=dict(size=7),
    ))
    fig_fold.update_layout(
        title="MAE by Validation Fold",
        xaxis_title="Target Year",
        yaxis_title="MAE (crimes per 100k)",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_fold, width="stretch")

with fold_right:
    hist = annual[annual["County"] == selected_county].sort_values("Year")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist["Year"], y=hist["Crime_Rate_per_100k"],
        name="Crime rate / 100k",
        mode="lines",
        fill="tozeroy",
        line=dict(color="#264653", width=2),
        yaxis="y1",
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist["Year"], y=hist["Population"],
        name="Population",
        mode="lines+markers",
        line=dict(color="#e76f51", width=2, dash="dot"),
        marker=dict(size=5),
        yaxis="y2",
    ))
    fig_hist.update_layout(
        title=f"Crime Rate & Population — {selected_county}",
        xaxis_title="Year",
        yaxis=dict(
            title=dict(text="Crime rate per 100k", font=dict(color="#264653")),
        ),
        yaxis2=dict(
            title=dict(text="Population", font=dict(color="#e76f51")),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_hist, width="stretch")

st.divider()

# ── county prediction snapshot ────────────────────────────────────────────────

st.subheader(f"Gradient Boosting predictions — {selected_county}")
county_snapshot = build_county_snapshot(preds_1yr, preds_2yr, selected_county)
if county_snapshot.empty:
    st.info("No prediction rows found for this county in the exported data.")
else:
    melt_cols = [c for c in ("Actual", "GB_Pred_1yr", "Actual_2yr", "GB_Pred_2yr") if c in county_snapshot.columns]
    chart_df  = county_snapshot.melt(
        id_vars=["County", "target_year"],
        value_vars=melt_cols,
        var_name="Series",
        value_name="Rate",
    ).dropna(subset=["Rate"])
    chart_df["Series Label"] = chart_df["Series"].map(
        lambda s: SERIES_LABELS.get(s, s)
    )
    fig_snapshot = px.line(
        chart_df, x="target_year", y="Rate", color="Series Label", markers=True,
        labels={"target_year": "Target Year", "Rate": "Crime rate per 100k", "Series Label": ""},
        title="Actual vs Predicted",
        color_discrete_map={v: SERIES_COLORS.get(k, "#999") for k, v in SERIES_LABELS.items()},
    )
    fig_snapshot.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_snapshot, width="stretch")
    with st.expander("Show raw prediction table"):
        st.dataframe(county_snapshot.round(2), width="stretch")

st.divider()

# ── county treemap ────────────────────────────────────────────────────────────

st.subheader("County population & crime rate — all counties")
st.caption("Click a region to drill down into its counties. Click the region label at the top to zoom back out.")
_tree_years = sorted(annual["Year"].dropna().unique().astype(int).tolist())
_default_tree_yr = _tree_years[-1] if _tree_years else 2022
tree_year = st.select_slider(
    "Select year",
    options=_tree_years,
    value=_default_tree_yr,
    key="tree_year_slider",
)
tree_df = annual[annual["Year"] == tree_year].dropna(
    subset=["Population", "Crime_Rate_per_100k", "Total_Crimes"]
).copy()
tree_df["Region"] = tree_df["County"].map(NY_COUNTY_REGION).fillna("Other")

# Rate of change vs prior year for the hover card
_prior = annual[annual["Year"] == tree_year - 1][["County", "Crime_Rate_per_100k"]].rename(
    columns={"Crime_Rate_per_100k": "Rate_Prior"}
)
tree_df = tree_df.merge(_prior, on="County", how="left")
tree_df["Rate_Change"] = tree_df["Crime_Rate_per_100k"] - tree_df["Rate_Prior"]
tree_df["Rate_Change_Str"] = tree_df["Rate_Change"].apply(
    lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}" if not pd.isna(v) else "n/a"
)

fig_tree = px.treemap(
    tree_df,
    path=["Region", "County"],
    values="Population",
    color="Crime_Rate_per_100k",
    color_continuous_scale="Reds",
    hover_name="County",
    custom_data=[
        "Crime_Rate_per_100k",
        "Population",
        "Total_Crimes",
        "Rate_Change_Str",
        "Region",
    ],
    labels={"Crime_Rate_per_100k": "Crime rate / 100k"},
    title=f"NY counties — {tree_year}   (box size = population · color = crime rate per 100k · click region to expand)",
)
fig_tree.update_traces(
    texttemplate=(
        "<b>%{label}</b><br>"
        "%{customdata[0]:.0f} / 100k"
    ),
    textposition="middle center",
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Region: %{customdata[4]}<br>"
        "Crime rate: %{customdata[0]:.1f} per 100k<br>"
        "Change vs prior year: %{customdata[3]}<br>"
        "Population: %{customdata[1]:,}<br>"
        "Total crimes: %{customdata[2]:,}"
        "<extra></extra>"
    ),
    marker=dict(
        line=dict(
            color=[
                "#e76f51" if c == selected_county else "#ffffff"
                for c in tree_df["County"]
            ],
            width=[4 if c == selected_county else 0.5 for c in tree_df["County"]],
        )
    ),
)
fig_tree.update_layout(
    height=560,
    margin=dict(t=50, l=5, r=5, b=5),
    coloraxis_colorbar=dict(title="per 100k"),
)
st.plotly_chart(fig_tree, width="stretch")
st.caption(f"Selected county **{selected_county}** is outlined in orange.")

# ── OpenStreetMap panel for selected county ───────────────────────────────────

import streamlit.components.v1 as components

_geo_lookup = COUNTY_NAME_MAP.get(selected_county, selected_county)
_centroid = NY_COUNTY_CENTROIDS.get(_geo_lookup) or NY_COUNTY_CENTROIDS.get(selected_county)

if _centroid:
    _lat, _lon = _centroid
    osm_url = (
        f"https://www.openstreetmap.org/export/embed.html"
        f"?bbox={_lon-0.35},{_lat-0.25},{_lon+0.35},{_lat+0.25}"
        f"&layer=mapnik"
        f"&marker={_lat},{_lon}"
    )
    st.markdown(f"**{selected_county} County — interactive map**")
    components.iframe(osm_url, height=340)
    st.caption(
        f"📍 Centered on {selected_county} County. "
        "Pan and zoom freely — click the ↗ icon to open in OpenStreetMap."
    )

st.divider()

# ── model leaderboard ─────────────────────────────────────────────────────────

st.subheader("Model leaderboard")
leaderboard = horizon_compare[["Model", "MAE_1yr", "MAE_2yr", "RMSE_1yr", "RMSE_2yr", "R2_1yr", "R2_2yr", "MAE_Gap"]].rename(
    columns={
        "MAE_1yr":  "MAE (1yr)",
        "MAE_2yr":  "MAE (2yr)",
        "RMSE_1yr": "RMSE (1yr)",
        "RMSE_2yr": "RMSE (2yr)",
        "R2_1yr":   "R² (1yr)",
        "R2_2yr":   "R² (2yr)",
        "MAE_Gap":  "Horizon Penalty",
    }
)
st.dataframe(leaderboard.round(3), width="stretch")

st.divider()

# ── rationale footer ──────────────────────────────────────────────────────────

with st.expander("Why recommend the 2-year model despite higher MAE?"):
    st.markdown(
        """
        **The 1-year model is more accurate** — that is expected, not a finding.
        Predicting one year closer to the present is an easier task: the lag features
        are fresher and there is less time for unexpected shocks to occur.

        **The 2-year model is more actionable** for law enforcement and policy use cases.
        Staffing decisions, budget allocations, and intervention planning require at
        least 18–24 months of lead time. A 1-year forecast arrives too late to act on.

        **Gradient Boosting still dominates at the harder horizon**, which validates it
        as the right model choice regardless of which horizon you deploy. The accuracy
        cost of forecasting further out is the *price of actionability*, not a model flaw.
        """
    )
