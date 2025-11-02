from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Config
# =========================
MAP_CENTER = dict(lat=58.5, lon=-106.3468)
MAP_ZOOM = 2.5
MAP_STYLE = "open-street-map"
CITY_COLOR = "#1f77b4"
CUST_COLOR = "#ff7f0e"

FIG_HEIGHT = 950
RIGHT_COL_WIDTH = 0.24
LEFT_COL_WIDTH = 1.0 - RIGHT_COL_WIDTH

TOPK_OPTIONS = [10, 25, 50]
DEFAULT_K = 25

PROVINCE_LABELS = [
    ("British Columbia", 53.7267, -127.6476),
    ("Alberta", 53.9333, -116.5765),
    ("Saskatchewan", 52.9399, -106.4509),
    ("Manitoba", 53.7609, -98.8139),
    ("Ontario", 50.0000, -85.0000),
    ("Quebec", 52.9399, -70.0000),
    ("New Brunswick", 46.5653, -66.4619),
    ("Nova Scotia", 45.0000, -62.9987),
    ("Prince Edward Island", 46.5107, -63.4168),
    ("Newfoundland & Labrador", 53.1355, -57.6604),
    ("Yukon", 63.0000, -135.0000),
    ("Northwest Territories", 64.8255, -124.8457),
    ("Nunavut", 66.0000, -96.0000),
]

# =========================
# Data utils
# =========================
def _safe_mode(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names
    if "Province" not in df.columns and "Province or State" in df.columns:
        df = df.rename(columns={"Province or State": "Province"})
    if "Customer Lifetime Value" in df.columns and "CLV" not in df.columns:
        df = df.rename(columns={"Customer Lifetime Value": "CLV"})

    # Coerce coords and drop invalid rows
    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Ensure numeric if present
    for c in ["CLV", "Income", "enroll_year", "cancel_year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure optional categoricals exist
    for c in ["Gender", "Education", "LoyaltyStatus", "Marital Status"]:
        if c not in df.columns:
            df[c] = np.nan

    # Ensure IsCancelled exists
    if "IsCancelled" not in df.columns:
        df["IsCancelled"] = 0

    return df

def _agg_city(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["City", "Province"], dropna=False)
          .agg(
              Customers=("Loyalty#", "count"),
              Avg_CLV=("CLV", "mean"),
              Cancel_Rate=("IsCancelled", lambda x: np.mean(pd.to_numeric(x, errors="coerce").fillna(0))),
              Lat=("Latitude", "mean"),
              Lon=("Longitude", "mean"),
              Gender_Mode=("Gender", _safe_mode),
              Education_Mode=("Education", _safe_mode),
              LoyaltyStatus_Mode=("LoyaltyStatus", _safe_mode),
              Marital_Mode=("Marital Status", _safe_mode),
          ).reset_index()
    )
    g["Cancel_Rate"] = g["Cancel_Rate"].round(3)
    return g

# =========================
# Trace builders
# =========================
def _province_labels_trace() -> go.Scattermapbox:
    lats = [lat for _, lat, _ in PROVINCE_LABELS]
    lons = [lon for _, _, lon in PROVINCE_LABELS]
    names = [name for name, _, _ in PROVINCE_LABELS]
    return go.Scattermapbox(
        lat=lats, lon=lons, mode="text", showlegend=False, hoverinfo="skip",
        text=names, textposition="middle center",
        textfont=dict(size=11, color="rgba(120,120,120,0.4)"),
        name="Province Labels", opacity=1.0
    )

def _city_trace(df_city: pd.DataFrame) -> go.Scattermapbox:
    if len(df_city):
        sizes = np.interp(df_city["Customers"], (df_city["Customers"].min(), df_city["Customers"].max()), (14, 44))
    else:
        sizes = []

    colors = df_city["Cancel_Rate"].values

    hover = (
        "<b>%{text}</b>, %{customdata[0]}<br>"
        "Customers: %{customdata[1]}<br>"
        "Avg CLV: %{customdata[2]:,.0f}<br>"
        "Cancel Rate: %{customdata[3]:.1%}<br>"
        "Most common gender: %{customdata[4]}<br>"
        "Most common education: %{customdata[5]}<br>"
        "Most common loyalty: %{customdata[6]}<br>"
        "Most common marital status: %{customdata[7]}<extra></extra>"
    )
    return go.Scattermapbox(
        lat=df_city["Lat"], lon=df_city["Lon"], mode="markers",
        name="Cities",
        marker=dict(
            size=sizes,
            opacity=0.85,
            color=colors,
            colorscale=[[0, '#2ecc71'], [0.5, '#f39c12'], [1, '#e74c3c']],
            cmin=0,
            cmax=df_city["Cancel_Rate"].max() if len(df_city) else 1,
            colorbar=dict(
                title="Cancel<br>Rate",
                thickness=12,
                len=0.25,
                x=1.02,
                y=0.15,
                xanchor="left",
                yanchor="middle",
                tickformat=".0%",
                titlefont=dict(size=10),
                tickfont=dict(size=9)
            )
        ),
        text=df_city["City"],
        customdata=np.stack([
            df_city["Province"], df_city["Customers"], df_city["Avg_CLV"], df_city["Cancel_Rate"],
            df_city["Gender_Mode"], df_city["Education_Mode"], df_city["LoyaltyStatus_Mode"], df_city["Marital_Mode"]
        ], axis=-1),
        hovertemplate=hover
    )

def _customer_trace(df: pd.DataFrame) -> go.Scattermapbox:
    # tamanho um pouco maior para legibilidade
    sizes = [10] * len(df)

    # JITTER: evita sobreposição total de clientes na mesma cidade
    jlat, jlon = _geo_jitter(df["Latitude"], df["Longitude"], meters=2000.0, seed=123)

    cancel_str = np.where(
        pd.to_numeric(df.get("IsCancelled", 0), errors="coerce").fillna(0).astype(int) == 1,
        "Cancelled", "Active"
    )

    hover = (
        "<b>%{text}</b><br>"
        "%{customdata[0]}, %{customdata[1]}<br>"
        "CLV: %{customdata[2]:,.0f}<br>"
        "Gender: %{customdata[3]}<br>"
        "Education: %{customdata[4]}<br>"
        "Status: %{customdata[5]}<extra></extra>"
    )

    return go.Scattermapbox(
        lat=jlat, lon=jlon, mode="markers",
        name="Customers",
        marker=dict(size=sizes, opacity=0.7, color=CUST_COLOR),
        text=df.get("Customer Name", df.get("Loyalty#", pd.Series([""] * len(df)))),
        customdata=np.stack([
            df["City"],
            df["Province"],
            df["CLV"].fillna(0),
            df.get("Gender", pd.Series(["Unknown"] * len(df))).fillna("Unknown"),
            df.get("Education", pd.Series(["Unknown"] * len(df))).fillna("Unknown"),
            pd.Series(cancel_str),
        ], axis=-1),
        hovertemplate=hover
    )


# def _customer_trace(df: pd.DataFrame) -> go.Scattermapbox:
#     sizes = [8] * len(df)
#     cancel_str = np.where(pd.to_numeric(df.get("IsCancelled", 0), errors="coerce").fillna(0).astype(int) == 1,
#                           "Cancelled", "Active")

#     hover = (
#         "<b>%{text}</b><br>"
#         "%{customdata[0]}, %{customdata[1]}<br>"
#         "CLV: %{customdata[2]:,.0f}<br>"
#         "Gender: %{customdata[3]}<br>"
#         "Education: %{customdata[4]}<br>"
#         "Status: %{customdata[5]}<extra></extra>"
#     )
#     return go.Scattermapbox(
#         lat=df["Latitude"], lon=df["Longitude"], mode="markers",
#         name="Customers", marker=dict(size=sizes, opacity=0.65, color=CUST_COLOR),
#         text=df.get("Customer Name", df.get("Loyalty#", pd.Series([""]*len(df)))),
#         customdata=np.stack([
#             df["City"],
#             df["Province"],
#             df["CLV"].fillna(0),
#             df.get("Gender", pd.Series(["Unknown"]*len(df))).fillna("Unknown"),
#             df.get("Education", pd.Series(["Unknown"]*len(df))).fillna("Unknown"),
#             pd.Series(cancel_str),
#         ], axis=-1),
#         hovertemplate=hover
#     )

# =========================
# Distributions (right column)
# =========================
def _city_distribution_traces(df: pd.DataFrame, title_key: str) -> List:
    """Return 5 traces (5 metrics: Gender, Education, Marital, Loyalty, Cancel) for the selection key."""
    if title_key == "__ALL__":
        sub = df.copy()
        title = "All Cities, Canada"
    else:
        try:
            city, prov = title_key.split("|||")
        except ValueError:
            city, prov = title_key, ""
        sub = df[(df["City"] == city) & (df["Province"] == prov)].copy()
        title = f"{city}, {prov}".strip(", ")

    total = len(sub)

    def pct(col: str) -> pd.Series:
        return sub[col].fillna("Unknown").value_counts(normalize=True).mul(100).round(1)

    gender = pct("Gender")
    education = pct("Education")
    marital = pct("Marital Status")
    loyal = pct("LoyaltyStatus")
    cancel_rate = pd.to_numeric(sub["IsCancelled"], errors="coerce").fillna(0).mean()*100.0

    traces = []

    # Gender distribution
    s = gender if not gender.empty else pd.Series({"No Data": 100.0})
    traces.append(go.Pie(labels=s.index.tolist(), values=s.values.tolist(), hole=0.4,
                       showlegend=False, name="Gender", textinfo="label+percent",
                       textfont=dict(size=9)))

    # Education distribution (right-biased labels)
    s = education if not education.empty else pd.Series({"No Data": 100.0})
    vals = np.asarray(s.values, dtype=float)
    vals = np.array([1.0]) if vals.sum() == 0 else vals
    angles = vals / vals.sum() * 360.0
    rotation = 90
    mid = (np.cumsum(angles) - angles/2.0 + rotation) % 360
    textpos = ["outside" if (0 <= a <= 90) or (270 <= a < 360) else "inside" for a in mid]

    traces.append(go.Pie(
        labels=s.index.tolist(),
        values=vals.tolist(),
        hole=0.55,
        showlegend=False,
        name="Education",
        textinfo="label+percent",
        textposition=textpos,
        insidetextorientation="radial",
        textfont=dict(size=8),
        rotation=rotation,
        sort=False,
        direction="clockwise"
    ))

    # Marital Status distribution
    s = marital if not marital.empty else pd.Series({"No Data": 100.0})
    traces.append(go.Pie(labels=s.index.tolist(), values=s.values.tolist(), hole=0.4,
                       showlegend=False, name="Marital", textinfo="label+percent",
                       textfont=dict(size=9)))

    # Loyalty distribution
    s = loyal if not loyal.empty else pd.Series({"No Data": 100.0})
    traces.append(go.Pie(labels=s.index.tolist(), values=s.values.tolist(), hole=0.4,
                       showlegend=False, name="Loyalty", textinfo="label+percent", textposition="auto",
                       insidetextorientation="radial",
                       textfont=dict(size=9)))

    # Cancel rate bar
    bar = go.Bar(x=["Cancelled"], text=[f"{cancel_rate:.1f}%"],
                 y=[cancel_rate],
                 name="Cancel", marker=dict(color='#e74c3c'),
                 textfont=dict(size=10), width=0.4, textposition="inside", insidetextanchor="middle")

    traces[0].hovertemplate = "%{label}: %{percent}<extra>Gender</extra>"
    traces[1].hovertemplate = "%{label}: %{percent}<extra>Education</extra>"
    traces[2].hovertemplate = "%{label}: %{percent}<extra>Marital Status</extra>"
    traces[3].hovertemplate = "%{label}: %{percent}<extra>Loyalty</extra>"
    bar.hovertemplate = "%{x}: %{y:.1f}%<extra>Cancel Rate</extra>"

    for t in traces + [bar]:
        t.meta = {"dist_key": title_key, "title": title, "n": total}

    return traces + [bar]

def _geo_jitter(lat: pd.Series, lon: pd.Series, meters: float = 2000.0, seed: int = 123):
    """
    Aplica um pequeno jitter (em metros) às coordenadas para evitar sobreposição de pontos.
    Mantém os pontos na vizinhança da cidade.
    """
    rng = np.random.default_rng(seed)
    lat = pd.to_numeric(lat, errors="coerce").astype(float).values
    lon = pd.to_numeric(lon, errors="coerce").astype(float).values

    # 1 grau de latitude ≈ 111_320 m; longitude depende de cos(lat)
    dlat_deg = rng.normal(loc=0.0, scale=meters / 111_320.0, size=len(lat))
    coslat = np.cos(np.deg2rad(np.clip(lat, -89.9, 89.9)))
    dlon_deg = rng.normal(
        loc=0.0,
        scale=meters / (111_320.0 * np.maximum(coslat, 1e-3)),
        size=len(lon)
    )

    return lat + dlat_deg, lon + dlon_deg


# =========================
# Main builder (public)
# =========================
def build_canada_customer_figure(customers_features_df: pd.DataFrame) -> go.Figure:
    # Prep
    DF = _clean_df(customers_features_df)
    DF_CITY = _agg_city(DF)

    # Add NumCustomers per (City, Province) back to DF for customer sizing
    city_counts = DF.groupby(["City","Province"], dropna=False).size().rename("NumCustomers").reset_index()
    DF = DF.merge(city_counts, on=["City","Province"], how="left")

    # Subplot layout
    specs = [
        [{"type": "mapbox", "rowspan": 5}, {"type": "domain"}],
        [None,                               {"type": "domain"}],
        [None,                               {"type": "domain"}],
        [None,                               {"type": "domain"}],
        [None,                               {"type": "xy"}],
    ]
    fig = make_subplots(
        rows=5, cols=2, specs=specs,
        column_widths=[LEFT_COL_WIDTH, RIGHT_COL_WIDTH],
        horizontal_spacing=0.00, vertical_spacing=0.015,
        subplot_titles=("", "", "", "", "")
    )
    fig.update_traces(selector=dict(col=2), domain_x=[LEFT_COL_WIDTH - 0.01, 1.0])

    # Province labels
    fig.add_trace(_province_labels_trace(), row=1, col=1)

    # City/Customer traces per Top-K
    city_traces_idx: Dict[int, int] = {}
    cust_traces_idx: Dict[int, int] = {}

    for k in TOPK_OPTIONS:
        # Top-K cities by Customers
        dfc = DF_CITY.sort_values("Customers", ascending=False).head(k)
        t_city = _city_trace(dfc)
        city_traces_idx[k] = len(fig.data)
        fig.add_trace(t_city, row=1, col=1)

        # Top-K customers by CLV
        top_idx = DF["CLV"].fillna(0).sort_values(ascending=False).index[:k]
        dfu = DF.loc[top_idx]
        t_cust = _customer_trace(dfu)
        cust_traces_idx[k] = len(fig.data)
        fig.add_trace(t_cust, row=1, col=1)

    # Right-side distributions for "__ALL__" + top max-k cities
    dist_keys = ["__ALL__"]
    top_cities_union = DF_CITY.sort_values("Customers", ascending=False).head(max(TOPK_OPTIONS))
    dist_keys += [f"{r.City}|||{r.Province}" for _, r in top_cities_union.iterrows()]

    dist_traces_indices: Dict[str, List[int]] = {}
    for key in dist_keys:
        traces = _city_distribution_traces(DF, key)
        idxs = []
        fig.add_trace(traces[0], row=1, col=2); idxs.append(len(fig.data)-1)
        fig.add_trace(traces[1], row=2, col=2); idxs.append(len(fig.data)-1)
        fig.add_trace(traces[2], row=3, col=2); idxs.append(len(fig.data)-1)
        fig.add_trace(traces[3], row=4, col=2); idxs.append(len(fig.data)-1)
        fig.add_trace(traces[4], row=5, col=2); idxs.append(len(fig.data)-1)
        dist_traces_indices[key] = idxs

    # Initial visibility
    vis = [True]  # province labels
    for k in TOPK_OPTIONS:
        vis.append(k == DEFAULT_K)  # city trace visible?
        vis.append(False)           # customer trace visible?
    for key in dist_keys:
        on = (key == "__ALL__")
        for _ in dist_traces_indices[key]:
            vis.append(on)

    fig.update_layout(
        mapbox=dict(style=MAP_STYLE, center=MAP_CENTER, zoom=MAP_ZOOM),
        height=FIG_HEIGHT,
        margin=dict(l=8, r=8, t=105, b=8),
        title=dict(text=f"Canada Customer Analytics • Top {DEFAULT_K} Cities",
                   x=0.01, xanchor="left", y=0.993, font=dict(size=14)),
        showlegend=False,
        paper_bgcolor="white", plot_bgcolor="white",
    )

    # Distribution section title
    fig.add_annotation(
        text="<b>Demographics: All Cities, Canada</b>",
        xref="paper", yref="paper",
        x=0.89, y=1.04, xanchor="center", yanchor="top",
        showarrow=False,
        font=dict(size=11, color="#2c3e50"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#bdc3c7",
        borderwidth=1,
        borderpad=4
    )

    # Subtitle fonts
    for i, annotation in enumerate(fig.layout.annotations[:-1]):
        annotation.font.size = 9
        annotation.font.color = "#34495e"

    for i, v in enumerate(vis):
        fig.data[i].visible = v

    # ---- shrink right-column donut domains vertically ----
    SHRINK_FACTOR = 0.9
    VSPACE = 0.00
    NROWS = 5

    def _row_y_domain(row_idx: int, nrows=NROWS, vspace=VSPACE):
        row_h = (1.0 - vspace*(nrows - 1)) / nrows
        y0 = 1.0 - (row_idx * row_h + (row_idx - 1) * vspace)
        y1 = y0 + row_h
        return y0, y1

    compressed_domains = {}
    for r in [1, 2, 3, 4]:
        y0, y1 = _row_y_domain(r)
        yc = (y0 + y1) / 2.0
        half = (y1 - y0) * SHRINK_FACTOR / 2.0
        compressed_domains[r] = [yc - half, yc + half]

    for r in [1, 2, 3, 4]:
        fig.update_traces(
            selector=dict(type="pie"),
            row=r, col=2,
            domain=dict(y=compressed_domains[r])
        )

    # Tight gap between columns
    TIGHT_GAP = 0.001
    RIGHT_PAD = 0.005
    start_x = LEFT_COL_WIDTH + TIGHT_GAP
    end_x = 1.0 - RIGHT_PAD
    for r in [1, 2, 3, 4, 5]:
        fig.update_traces(
            selector=dict(col=2, row=r),
            domain_x=[start_x, end_x]
        )

    # ---- helpers to compute visibility for controls ----
    base_count = len(fig.data)

    def _vis_arrays(mode: str, k: int, dist_key: str):
        out = [False] * base_count
        out[0] = True  # province labels
        # city/customer per k
        for i, kk in enumerate(TOPK_OPTIONS):
            city_idx = 1 + 2*i
            cust_idx = 2 + 2*i
            if mode == "city":
                out[city_idx] = (kk == k)
                out[cust_idx] = False
            elif mode == "customer":
                out[city_idx] = False
                out[cust_idx] = (kk == k)
        # distributions
        dist_start = 1 + 2*len(TOPK_OPTIONS)
        cursor = dist_start
        for key in dist_keys:
            on = (key == dist_key)
            for _ in dist_traces_indices[key]:
                out[cursor] = on
                cursor += 1
        return out

    # =========================
    # Controls
    # =========================

    # Botões de modo (vão alternar também a visibilidade dos menus Top-K e dropdowns)
    mode_buttons = [
        dict(
            label="Cities View",
            method="update",
            args=[
                {"visible": _vis_arrays("city", DEFAULT_K, "__ALL__")},
                {
                    "title.text": f"Canada Customer Analytics • Top {DEFAULT_K} Cities",
                    "annotations[-1].text": "<b>Demographics: All Cities, Canada</b>",
                    # Torna visível TopK Cities [1] e Dropdown Cities [3]; esconde os de Customers [2] e [4]
                    "updatemenus[1].visible": True,
                    "updatemenus[2].visible": False,
                    "updatemenus[3].visible": True,
                    "updatemenus[4].visible": False,
                }
            ],
        ),
        dict(
            label="Customers View",
            method="update",
            args=[
                {"visible": _vis_arrays("customer", DEFAULT_K, "__ALL__")},
                {
                    "title.text": f"Canada Customer Analytics • Top {DEFAULT_K} Customers",
                    "annotations[-1].text": "<b>Demographics: All Cities, Canada</b>",
                    "mapbox.zoom": 3.2,
                    "updatemenus[1].visible": False,
                    "updatemenus[2].visible": True,
                    "updatemenus[3].visible": False,
                    "updatemenus[4].visible": True,
                }
            ],
        ),
    ]

    # Top-K para Cities (age sobre modo city)
    topk_buttons_city = [
        dict(
            label=f"Top {k}",
            method="update",
            args=[
                {"visible": _vis_arrays("city", k, "__ALL__")},
                {
                    "title.text": f"Canada Customer Analytics • Top {k} Cities",
                    "annotations[-1].text": "<b>Demographics: All Cities, Canada</b>"
                }
            ],
        )
        for k in TOPK_OPTIONS
    ]

    # Top-K para Customers (age sobre modo customer)
    topk_buttons_cust = [
        dict(
            label=f"Top {k}",
            method="update",
            args=[
                {"visible": _vis_arrays("customer", k, "__ALL__")},
                {
                    "title.text": f"Canada Customer Analytics • Top {k} Customers",
                    "annotations[-1].text": "<b>Demographics: All Cities, Canada</b>"
                }
            ],
        )
        for k in TOPK_OPTIONS
    ]

    # Dropdowns de cidade — um para cada modo
    city_buttons_citymode = []
    city_buttons_custmode = []
    for key in dist_keys:
        if key == "__ALL__":
            label = "All Cities"
            title_main_city = "Canada Customer Analytics • All Cities"
            title_main_cust = "Canada Customer Analytics • All Cities"
            title_demo = "<b>Demographics: All Cities, Canada</b>"
        else:
            city, prov = key.split("|||")
            label = f"{city}, {prov}"
            title_main_city = f"Canada Customer Analytics • {city}, {prov}"
            title_main_cust = f"Canada Customer Analytics • {city}, {prov}"
            title_demo = f"<b>Demographics: {city}, {prov}</b>"

        # dropdown que mantém o modo City
        city_buttons_citymode.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": _vis_arrays("city", DEFAULT_K, key)},
                    {"title.text": title_main_city, "annotations[-1].text": title_demo},
                ],
            )
        )
        # dropdown que mantém o modo Customer
        city_buttons_custmode.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": _vis_arrays("customer", DEFAULT_K, key)},
                    {"title.text": title_main_cust, "annotations[-1].text": title_demo},
                ],
            )
        )

    # Layout dos controlos (5 menus):
    # [0] Modo, [1] TopK Cities, [2] TopK Customers, [3] Dropdown Cities (City mode),
    # [4] Dropdown Cities (Customer mode)
    fig.update_layout(
        updatemenus=[
            dict(  # [0] Mode
                type="buttons",
                direction="left",
                buttons=mode_buttons,
                x=0.01, xanchor="left", y=1.085, yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                active=0,
                bordercolor="#34495e", borderwidth=1,
                font=dict(size=11, color="#34495e"),
                pad={"r":5,"t":5,"b":5,"l":5},
                visible=True
            ),
            dict(  # [1] Top-K (Cities)
                type="buttons",
                direction="left",
                buttons=topk_buttons_city,
                x=0.28, xanchor="left", y=1.085, yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                active=TOPK_OPTIONS.index(DEFAULT_K),
                bordercolor="#16a085", borderwidth=1,
                font=dict(size=11, color="#16a085"),
                pad={"r":5,"t":5,"b":5,"l":5},
                visible=True   # começa em Cities
            ),
            dict(  # [2] Top-K (Customers)
                type="buttons",
                direction="left",
                buttons=topk_buttons_cust,
                x=0.28, xanchor="left", y=1.085, yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                active=TOPK_OPTIONS.index(DEFAULT_K),
                bordercolor="#16a085", borderwidth=1,
                font=dict(size=11, color="#16a085"),
                pad={"r":5,"t":5,"b":5,"l":5},
                visible=False  # oculto até mudares para Customers
            ),
            dict(  # [3] Dropdown (City mode)
                type="dropdown",
                direction="down",
                buttons=city_buttons_citymode,
                x=0.60, xanchor="left", y=1.085, yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                active=0,
                bordercolor="#7f8c8d", borderwidth=1,
                font=dict(size=10),
                pad={"r":5,"t":5,"b":5,"l":5},
                visible=True
            ),
            dict(  # [4] Dropdown (Customer mode)
                type="dropdown",
                direction="down",
                buttons=city_buttons_custmode,
                x=0.60, xanchor="left", y=1.085, yanchor="top",
                bgcolor="rgba(255,255,255,0.95)",
                active=0,
                bordercolor="#7f8c8d", borderwidth=1,
                font=dict(size=10),
                pad={"r":5,"t":5,"b":5,"l":5},
                visible=False
            ),
        ]
    )

    # Pequeno ajuste ao centro do mapa (como tinhas)
    MAP_CENTER_SHIFT_LAT = -19.0
    MAP_CENTER_SHIFT_LON = 13.0
    fig.update_layout(
        mapbox=dict(
            center=dict(
                lat=MAP_CENTER["lat"] + MAP_CENTER_SHIFT_LAT,
                lon=MAP_CENTER["lon"] + MAP_CENTER_SHIFT_LON
            )
        )
    )

    return fig
