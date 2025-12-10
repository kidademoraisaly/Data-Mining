
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd 
import plotly.graph_objects as go
import re
from sklearn.base import clone
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib import cm
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from pandas.plotting import parallel_coordinates
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

import pandas as pd
def plot_histograms(df, columns, n_rows=2, title="Numeric Variables' Histograms", bins=20):
    # Grid layout
    sns.set_style("white")
    figsize=(20, 11)
    n_cols = max(1, ceil(len(columns) / max(1, n_rows)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).ravel()  # flatten safely for any shape

    # Plot each histogram
    for ax, feat in zip(axes, columns):
        ax.hist(df[feat].dropna(), bins=bins)
        ax.set_title(str(feat), y=-0.13)

    # Hide unused axes
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_daily_trends(
    df,
    date_cols,
    labels=None,
    title="Daily Events Over Time",
    spike_threshold=2,
    
):
    """
    Interactive Plotly time series for daily event counts with spike highlighting.

    """
    df = df.copy()

    fig = go.Figure()

    for i, col in enumerate(date_cols):
        label = labels[i] if labels else col

        # Count daily occurrences
        daily_counts = df[col].value_counts().sort_index()

        # Detect spikes (days with unusually high counts)
        threshold = daily_counts.median() * spike_threshold
        spikes = daily_counts[daily_counts > threshold]

        # Main line
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts.values,
            mode="lines+markers",
            name=label,
        ))

        # Spike markers
        if not spikes.empty:
            fig.add_trace(go.Scatter(
                x=spikes.index,
                y=spikes.values,
                mode="markers",
                marker=dict(color="red", size=10, symbol="star"),
                name=f"{label} Spikes (> {spike_threshold}× median)"
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        legend=dict(x=0.5, y=-0.2, orientation="h")
    )

    fig.show()


def plot_histograms(df, columns, n_rows=2, title="Numeric Variables' Histograms", bins=20, max_unique_discrete=20,    figsize = (20, 11)):
    """
    Plots histograms for numeric columns.
    - If a column has fewer than `max_unique_discrete` unique values or is integer dtype,
      it is treated as discrete (one bin per value).
    - Otherwise, a regular histogram is used.
    """
    sns.set_style("white")

    n_cols = max(1, ceil(len(columns) / max(1, n_rows)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).ravel()

    for ax, feat in zip(axes, columns):
        s = df[feat].dropna()

        # Determine if the column is discrete
        is_discrete = (
            np.issubdtype(s.dtype, np.integer)
            and s.nunique() <= max_unique_discrete
        )

        if is_discrete:
            # One bin per unique value
            vals = np.sort(s.unique())
            bins_disc = np.arange(vals.min() - 0.5, vals.max() + 1.5, 1)
            ax.hist(s, bins=bins_disc)
            ax.set_xticks(vals)
        else:
            # Regular continuous histogram
            ax.hist(s, bins=bins)

        #ax.set_title(str(feat), y=-0.13)
        ax.set_xlabel(str(feat))

    # Hide unused axes
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.97])
    plt.show()



def plot_boxplots(df, columns, n_rows=2, title="Numeric Variables' Box Plots", figsize = (20, 11)):
    # Prepare figure: n_rows rows, enough columns to fit all features
    sns.set_style("white")
   
    n_cols = max(1, ceil(len(columns) / max(1, n_rows)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).ravel()  # flatten safely for any shape

    # Plot data
    for ax, feat in zip(axes, columns):
        sns.boxplot(x=df[feat].dropna(), ax=ax)
        #ax.set_title(feat, y=-0.13)

    # Hide any unused axes
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    # Layout
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def pairplot(df, columns, col_to_analyse=None, n_rows=2):
    if col_to_analyse is not None:
        if col_to_analyse not in columns:
            raise ValueError(f"'{col_to_analyse}' not found in provided columns list.")

        others = [c for c in columns if c != col_to_analyse]
        n = len(others)
        n_cols = (n + n_rows - 1) // n_rows  # dynamically compute number of columns

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(others):
            sns.scatterplot(x=df[col_to_analyse], y=df[col], ax=axes[i])
            axes[i].set_title(f"{col_to_analyse} vs {col}")
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f"Scatterplots of '{col_to_analyse}' vs Other Variables", fontsize=18, y=1.02)
        plt.show()

    else:
        # Full pairplot if no focus variable is provided
        sns.pairplot(df[columns], diag_kind="hist")
        plt.subplots_adjust(top=0.95)
        plt.suptitle("Pairwise Relationship of Numerical Variables", fontsize=20)
        plt.show()

def plot_monthly_seasonality(desc_df, prefix=None, title="Monthly Seasonality Trend"):
    """
    Plots monthly seasonality (mean, 25%, 50%, 75%) from a .describe().T DataFrame
    produced from the output of compute_monthly_share_trends().
    """
    # 1) Filter rows that correspond to monthly columns, e.g. "Flights_Month_7"
    idx = desc_df.index.astype(str)
    if prefix or  prefix=="":
        month_rows = [name for name in idx if name.startswith(f"{prefix}")]
    else:
        # if columns are just month numbers (e.g., '1', '2', '3', ...)
        month_rows = [int(name) for name in idx if re.fullmatch(r"\d{1,2}", name)]

    if not month_rows:
        raise ValueError(
            f"No rows found in desc_df.index matching pattern for prefix '{prefix}'. "
            "Check the DataFrame or prefix used in .describe().T."
        )

    d = desc_df.loc[month_rows].copy()

    # 2) Extract month number from the end of the name (e.g. "Flights_Month_12" → 12)
    def _extract_month(name) -> int:
        if type(name)==int:
            return name
        m = re.search(r'(\d{1,2})$', name)
        if not m:
            raise ValueError(f"Could not extract month number from: {name}")
        return int(m.group(1))

    d["Month"] = [_extract_month(n) for n in d.index]
    d = d.sort_values("Month")

    # 3) Month names for x-axis
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    # 4) Plot
    plt.figure(figsize=(10, 6))
    plt.plot(d["Month"], d["mean"], marker="o", linewidth=2, label="Mean")
    plt.plot(d["Month"], d["25%"], linestyle="--", label="25th percentile")
    plt.plot(d["Month"], d["50%"], linestyle="-.", label="Median (50%)")
    plt.plot(d["Month"], d["75%"], linestyle="--", label="75th percentile")

    # Interquartile Range (IQR) band
    plt.fill_between(d["Month"], d["25%"], d["75%"],
                     alpha=0.1, color="gray", label="IQR (25–75%)")

    plt.title(title, fontsize=13)
    plt.xlabel("Month")
    plt.ylabel("Share (%)")
    plt.xticks(range(1, 13), month_names)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_seasonal_seasonality(
    desc_df,
    prefix="PropCompFlightsSeason_",
    title="Seasonal Companion Flights Trend"
):
    """
    Plot seasonal seasonality (mean, 25%, 50%, 75%) from a .describe().T DataFrame
    produced from compute_seasonal_share_trends().
    """

    # 1) pick rows whose index starts with the given prefix
    idx = desc_df.index.astype(str)
    season_rows = [name for name in idx if name.startswith(prefix)]
    if not season_rows:
        raise ValueError(
            f"No rows in desc_df.index match prefix '{prefix}'. "
            "Did you run .describe().T on the seasonal features?"
        )
    d = desc_df.loc[season_rows].copy()

    # 2) extract season key after the prefix (e.g., 'PropSeason_LowSeasonEarlyYear' -> 'LowSeasonEarlyYear')
    d["SeasonKey"] = [str(n)[len(prefix):] for n in d.index]

    # 3) map to display labels and fixed order
    key_to_label = {
        "LowSeasonEarlyYear": "Low Season (Early Year)",
        "PeakSeasonSummer": "Peak Season (Summer)",
        "Autumn": "Autumn",
        "HolidayPeak": "Holiday Peak",
    }
    season_order = ["LowSeasonEarlyYear", "PeakSeasonSummer", "Autumn", "HolidayPeak"]

    # keep only known keys and order them
    d = d[d["SeasonKey"].isin(season_order)].copy()
    d["Order"] = d["SeasonKey"].map({k: i for i, k in enumerate(season_order)})
    d = d.sort_values("Order")
    labels = d["SeasonKey"].map(key_to_label).tolist()

    # 4) plot
    x = range(len(d))
    plt.figure(figsize=(8, 5))
    plt.plot(x, d["mean"], marker="o", linewidth=2, label="Mean")
    plt.plot(x, d["25%"], linestyle="--", label="25th percentile")
    plt.plot(x, d["50%"], linestyle="-.", label="Median (50%)")
    plt.plot(x, d["75%"], linestyle="--", label="75th percentile")
    plt.fill_between(x, d["25%"], d["75%"], alpha=0.1, label="IQR (25–75%)")

    plt.title(title)
    plt.xlabel("Season")
    plt.ylabel("Share (%)")
    plt.xticks(x, labels, rotation=0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_countplots(
    df, columns, n_rows=2, title="Categorical Variables' Absolute Counts",
    figsize=(20, 11), top_k=None, min_freq=None, other_label="Other",
    horizontal=False, rotate=0
):
    sns.set_style("white")
    n_cols = max(1, ceil(len(columns) / max(1, n_rows)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).ravel()

    for ax, feat in zip(axes, columns):
        s = df[feat].astype("object")
        vc = s.value_counts(dropna=False)
        uniq = len(vc)

        # sensible default for very wide categories
        tk = top_k
        mf = min_freq
        if tk is None and mf is None and uniq > 30:
            tk = 20

        # group infrequent categories into "Other"
        if tk is not None:
            keep = set(vc.head(tk).index)
            s = s.where(s.isin(keep), other_label)
        elif mf is not None:
            keep = set(vc[vc >= mf].index)
            s = s.where(s.isin(keep), other_label)

        order = s.value_counts().index

        if horizontal:
            sns.countplot(y=s, order=order, ax=ax)
        else:
            sns.countplot(x=s, order=order, ax=ax)
            if rotate:
                ax.tick_params(axis="x", labelrotation=rotate)

       

    for ax in axes[len(columns):]:
        ax.set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    #return fig, axes


def plot_bivariate_relative(df, cols, figsize=(15, 20), n_rows=1, top_k=10, col_to_analyse=None):
    """
    Plot relative stacked bar charts for combinations of categorical columns.
    If col_to_analyse is provided, only plot combinations involving that column.
    Keeps only top_k most frequent categories for each variable, grouping the rest as 'Other'.
    """
    
    # Determine column combinations
    if col_to_analyse is not None:
        # Only include combos where col_to_analyse appears
        combos = [(col_to_analyse, c) for c in cols if c != col_to_analyse]
    else:
        # All pairwise combinations
        combos = list(itertools.combinations(cols, 2))
    
    n_plots = len(combos)
    if n_plots == 0:
        print("No valid combinations to plot.")
        return
    
    n_cols = (n_plots + n_rows - 1) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, (x, y) in enumerate(combos):
        top_x = df[x].value_counts().nlargest(top_k).index
        top_y = df[y].value_counts().nlargest(top_k).index

        df_copy = df.copy()
        df_copy[x] = df_copy[x].where(df_copy[x].isin(top_x), "Other")
        df_copy[y] = df_copy[y].where(df_copy[y].isin(top_y), "Other")

        pd.crosstab(df_copy[x], df_copy[y], normalize='index').plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'{x} vs {y} (Relative counts)')
        axes[i].legend(loc=(1.02, 0))
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_bivariate_absolute(df, cols, figsize=(12, 4), n_rows=1, top_k=10):
    """
    Plot absolute stacked bar charts for all combinations of categorical columns.
    Keeps only top_k most frequent categories for each variable, grouping the rest as 'Other'.
    """
    combos = list(itertools.combinations(cols, 2))
    n_plots = len(combos)
    n_cols = (n_plots + n_rows - 1) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, (x, y) in enumerate(combos):
        top_x = df[x].value_counts().nlargest(top_k).index
        top_y = df[y].value_counts().nlargest(top_k).index

        df_copy = df.copy()
        df_copy[x] = df_copy[x].where(df_copy[x].isin(top_x), "Other")
        df_copy[y] = df_copy[y].where(df_copy[y].isin(top_y), "Other")

        pd.crosstab(df_copy[x], df_copy[y]).plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'{x} vs {y} (Absolute counts)')
        axes[i].legend(loc=(1.02, 0))
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()



def plot_numeric_vs_categorical(df, numeric_cols, hue, figsize=(20, 10), n_rows=2, bins=10, top_k=10,   stat="count",  multiple="stack" ):
    """
    Plots histograms of numeric variables colored by a categorical hue variable.
    Keeps only the top K_top hue categories; the rest are grouped as 'Other'.
    """
    # Work on a copy so we don't mutate the original df
    df_copy = df.copy()

    # Keep top-K hue categories, group the rest as 'Other'
    top_hue_values = df_copy[hue].value_counts().nlargest(top_k).index
    df_copy[hue] = df_copy[hue].where(df_copy[hue].isin(top_hue_values), "Other")

    n_plots = len(numeric_cols)
    n_cols = (n_plots + n_rows - 1) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, feat in zip(axes, numeric_cols):
        sns.histplot(df_copy, x=feat, bins=bins, hue=hue, ax=ax, multiple=multiple, stat=stat, kde=False)
        ax.set_title(f'{feat} by {hue}')

    # Hide any unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Numeric Variables' Histograms by {hue}", fontsize=16)
    #plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    plt.show()



def plot_pairplot_with_categorical(
    df, numeric_cols, hue,
    top_k=10,
    diag_kind="hist",      # "hist" or "kde"
    plot_kind="scatter",   # "scatter" or "kde"
    height=2.5,            # size of each subplot
    alpha=0.6              # transparency for scatterplots
):
    """
    Creates a Seaborn pairplot showing pairwise relationships among numeric variables,
    colored by a categorical variable (hue).

    """
    df_copy = df.copy()

    # Keep top K categories for hue
    top_hue_values = df_copy[hue].value_counts().nlargest(top_k).index
    df_copy[hue] = df_copy[hue].where(df_copy[hue].isin(top_hue_values), "Other")

    # Select relevant columns
    cols_to_plot = numeric_cols + [hue]

    # Create the pairplot
    g = sns.pairplot(
        df_copy[cols_to_plot],
        hue=hue,
        diag_kind=diag_kind,
        kind=plot_kind,
        height=height,
        plot_kws={"alpha": alpha}
    )

    # Add title and spacing
    plt.suptitle("Pairwise Relationship of Numerical Variables", fontsize=20)
    plt.subplots_adjust(top=0.93)
    plt.show()


def compute_monthly_share_trends(
    flights_df: pd.DataFrame,
    value_col: str,
    prefix: str,
    total_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute monthly percentages per customer and average them across years.
    """
    # 1) Aggregate to monthly per customer-year
    group_keys = ["Loyalty#", "Year", "Month"]
    agg_cols = [value_col] if total_col is None else [value_col, total_col]
    df_month = (
        flights_df
        .groupby(group_keys, as_index=False)[agg_cols]
        .sum()
    )

    if total_col is None:
        # --- Year-normalized share (original behavior) ---
        year_totals = (
            df_month.groupby(["Loyalty#", "Year"], as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: "YearTotal"})
        )
        df = df_month.merge(year_totals, on=["Loyalty#", "Year"], how="left")
        df["YearTotal"] = df["YearTotal"].replace(0, np.nan)
        df["MonthlyShare_pct"] = (df[value_col] / df["YearTotal"]) * 100.0

    else:
        # --- Monthly ratio: numerator=value_col, denominator=total_col (per month) ---
        df = df_month.copy()
        df[total_col] = df[total_col].replace(0, np.nan)
        df["MonthlyShare_pct"] = (df[value_col] / df[total_col]) * 100.0

    # 2) Average percentages across years per (customer, month)
    cust_month_avg = (
        df.groupby(["Loyalty#", "Month"], as_index=False)["MonthlyShare_pct"]
          .mean()
          .rename(columns={"MonthlyShare_pct": "MonthlyShare_pct_avg"})
    )

    # 3) Pivot to wide (one column per month)
    wide = cust_month_avg.pivot_table(
        index="Loyalty#",
        columns="Month",
        values="MonthlyShare_pct_avg",
        fill_value=0.0
    )

    # Clean up: ensure 12 month columns, remove axis name, rename with prefix
    wide = wide.rename_axis(None, axis=1)
    wide = wide.reindex(columns=range(1, 13), fill_value=0.0)
    wide = wide.rename(columns={m: f"{prefix}Month_{m}" for m in range(1, 13)})

    return wide.reset_index().round(2)

def get_columns_with_prefix(columns, prefix="Flights_", months=None):
    """
    Return column names matching a given prefix and optional list of months.
    Example match: 'Flights_Month_1'
    """
    pattern = rf"^{re.escape(prefix)}(\d{{1,2}})$"
    matched = []
    for col in map(str, columns):
        m = re.match(pattern, col)
        if m and (months is None or int(m.group(1)) in months):
            matched.append(col)
    return matched



def _month_to_season(m: int) -> str:
    """Map month number to descriptive season group."""
    if 1 <= m <= 5:
        return "Low Season (Early Year)"
    elif 6 <= m <= 8:
        return "Peak Season (Summer)"
    elif 9 <= m <= 11:
        return "Autumn"
    elif m == 12:
        return "Holiday Peak"
    else:
        return "Unknown"
    

def add_peak_based_season_shares(
    df: pd.DataFrame,
    month_prefix: str = "Flights_Month_",
    keep_intermediate: bool = True,
) -> pd.DataFrame:
    """
    Add peak-based seasonal share features to a wide flights-per-month dataframe.

    Assumptions:
      - df has columns: Flights_Month_1, ..., Flights_Month_12 (or a different prefix).
      - Each row corresponds to a customer (or customer-year) and contains monthly flights.
      - We only use the maximum *value* in each season, not the month index.

    Steps:
      1. Define four seasons:
         - LowSeasonEarlyYear: months 1, 2, 3, 5, 6
         - PeakSeasonSummer:   months 7, 8
         - Autumn:             months 9, 10, 11
         - HolidayPeak:        month 12
      2. For each season, compute the row-wise maximum flights within that season:
         Max_Season = max over [Flights_Month_m for m in season_months]
      3. Compute YearPeakTotal = sum of the four Max_Season values.
      4. Compute SeasonShareInYear_Season = Max_Season / YearPeakTotal * 100.
      5. Return the transformed dataframe.
    """

    out = df.copy()
    print("Reading---")
    # 1. Define months in each season
    season_months = {
        "LowSeasonEarlyYear": [1, 2, 3, 4, 5],   # Jan, Feb, Mar, May, Jun
        "PeakSeasonSummer":   [6, 7, 8],            # Jul, Aug
        "Autumn":             [9, 10, 11],       # Sep, Oct, Nov
        "HolidayPeak":        [12],              # Dec
    }

    # 2. Compute the maximum number of flights in each season (row-wise max VALUE)
    for season, months in season_months.items():
        month_cols = [
            f"{month_prefix}{m}"
            for m in months
            if f"{month_prefix}{m}" in out.columns
        ]

        if not month_cols:
            # No matching columns found for this season (edge case)
            out[f"Max_{season}"] = 0.0
        else:
            # Ensure numeric, then take max *value* across the season months
            out[month_cols] = out[month_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            out[f"Max_{season}"] = out[month_cols].max(axis=1)
            out[f"columns_{season}"]= ",".join(month_cols)

    # 3. Total "peak flights" across all seasons
    peak_cols = [f"Max_{s}" for s in season_months.keys()]
    out["YearPeakTotal"] = out[peak_cols].sum(axis=1)

    # 4. Avoid division by zero (no flights at all)
    out["YearPeakTotal"] = out["YearPeakTotal"].replace(0, np.nan)

    # 5. Compute peak-based seasonal share within the year (in %)
    share_cols = []
    for season in season_months.keys():
        share_col = f"SeasonShareInYear_{season}"
        out[share_col] = (out[f"Max_{season}"] / out["YearPeakTotal"]) * 100.0
        share_cols.append(share_col)

    # Replace NaNs (e.g. customers with no flights) by 0 in the new columns
    out[["YearPeakTotal"] + share_cols] = out[["YearPeakTotal"] + share_cols].fillna(0.0)

    # Optionally drop intermediate helper columns
    if not keep_intermediate:
        out = out.drop(columns=peak_cols + ["YearPeakTotal"])

    return out


def compute_seasonal_share_trends(
    flights_df,
    value_col,
    total_col,
    prefix: str = "PropOfCompSeason_",
) :
    """
    Compute seasonal % per customer by:
      - summing value/total within each season for each (Loyalty#, Year)
      - computing share = value_sum / total_sum * 100
      - averaging the share across years per (Loyalty#, SeasonGroup)
      - pivoting to wide with ordered season columns
    """
    df = flights_df.copy()

    # ensure Month column exists (1..12)
    if "Month" not in df.columns and "YearMonthDate" in df.columns:
        df["Month"] = df["YearMonthDate"].dt.month

    # assign season group
    df["SeasonGroup"] = df["Month"].astype(int).map(_month_to_season)

    # aggregate within (Loyalty#, Year, SeasonGroup)
    group_keys = ["Loyalty#", "Year", "SeasonGroup"]
    agg = df.groupby(group_keys, as_index=False)[[value_col, total_col]].sum()

    # share per year-season
    agg[total_col] = agg[total_col].replace(0, np.nan)
    agg["SeasonShare_pct"] = (agg[value_col] / agg[total_col]) * 100.0

    # average across years per (customer, season)
    cust_season_avg = (
        agg.groupby(["Loyalty#", "SeasonGroup"], as_index=False)["SeasonShare_pct"]
           .mean()
           .rename(columns={"SeasonShare_pct": "SeasonShare_pct_avg"})
    )

    # pivot to wide with fixed season order
    season_order = [
        "Low Season (Early Year)",
        "Peak Season (Summer)",
        "Autumn",
        "Holiday Peak",
    ]
    wide = cust_season_avg.pivot_table(
        index="Loyalty#", columns="SeasonGroup", values="SeasonShare_pct_avg"
    )

    # enforce all seasons and fill missing values
    wide = wide.reindex(columns=season_order).fillna(0.0)

    # rename columns with prefix
    wide = wide.rename(
        columns={s: f"{prefix}{s.replace(' ', '').replace('(', '').replace(')', '')}" for s in wide.columns}
    )

    return wide.reset_index().round(2)


def get_ss(df, feats):
    """
    Calculate the sum of squares (SS) for the given DataFrame.

    The sum of squares is computed as the sum of the variances of each column
    multiplied by the number of non-NA/null observations minus one.

    Parameters:
    df (pandas.DataFrame): The input DataFrame for which the sum of squares is to be calculated.
    feats (list of str): A list of feature column names to be used in the calculation.

    Returns:
    float: The sum of squares of the DataFrame.
    """
    df_ = df[feats]
    ss = np.sum(df_.var() * (df_.count() - 1))
    
    return ss 


def get_ssb(df, feats, label_col):
    """
    Calculate the between-group sum of squares (SSB) for the given DataFrame.
    The between-group sum of squares is computed as the sum of the squared differences
    between the mean of each group and the overall mean, weighted by the number of observations
    in each group.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column in the DataFrame that contains the group labels.
    
    Returns
    float: The between-group sum of squares of the DataFrame.
    """
    
    ssb_i = 0
    for i in np.unique(df[label_col]):
        df_ = df.loc[:, feats]
        X_ = df_.values
        X_k = df_.loc[df[label_col] == i].values
        
        ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))) )

    ssb = np.sum(ssb_i)
    

    return ssb


def get_ssw(df, feats, label_col):
    """
    Calculate the sum of squared within-cluster distances (SSW) for a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing cluster labels.

    Returns:
    float: The sum of squared within-cluster distances (SSW).
    """
    feats_label = feats+[label_col]

    df_k = df[feats_label].groupby(by=label_col).apply(
        lambda col: get_ss(col, feats), 
        include_groups=False
        )

    return df_k.sum()

def get_rsq(df, feats, label_col):
    """
    Calculate the R-squared value for a given DataFrame and features.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    feats (list): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing the labels or cluster assignments.
    
    Returns:
    float: The R-squared value, representing the proportion of variance explained by the clustering.
    """
    df_sst_ = get_ss(df, feats)  # get total sum of squares
    df_ssw_ = get_ssw(df, feats, label_col)  # get ss within
    df_ssb_ = df_sst_ - df_ssw_  # get ss between
    # r2 = ssb/sst
    return (df_ssb_ / df_sst_)


def get_r2_scores(df, feats, clusterer, min_k=1, max_k=9):
    """
    Loop over different values of k. To be used with sklearn clusterers.
    """
    r2_clust = {}
    for n in range(min_k, max_k):
        clust = clone(clusterer).set_params(n_clusters=n)
        labels = clust.fit_predict(df)
        df_concat = pd.concat([df,
                               pd.Series(labels, name='labels', index=df.index)], axis=1)
        r2_clust[n] = get_rsq(df_concat, feats, 'labels')
    return r2_clust


def compute_r2_scores(
    df,
    feature_cols,
    param_values,
    cluster_factory,
    param_name="param",
 
):
    """
    Computes R2 scores (or any custom scoring metric) for a range of clustering parameter values.
    """
    #results = {}
    results=[]
    for val in param_values:
        clusterer = cluster_factory(val)

        # Fit clustering
        if hasattr(clusterer, "fit_predict"):
            labels = clusterer.fit_predict(df[feature_cols])
        else:
            clusterer.fit(df[feature_cols])
            labels = clusterer.labels_

        df_concat = pd.concat(
            [df[feature_cols], pd.Series(labels, name="labels", index=df.index)],
            axis=1
        )
        
        score = get_rsq(df_concat, feature_cols, "labels")
        result={}
        result[param_name]=val
        result["score"] = score
        results.append(result)

        print(f"{param_name}={val} -> score: {score:.4f}")
    return pd.DataFrame(results)
    #return pd.DataFrame(results,index=[0]).T
    #return pd.DataFrame(results, index=[0]).T.rename(columns={0: param_name,1: "score"})


def plot_r2_scores(r2_df, param_name="Parameter", figsize=(10,6), title=None):
    """
    Plots the R² scores returned by compute_r2_scores.
    """
    plt.figure(figsize=figsize)
    plt.plot(r2_df.index, r2_df['score'], marker='o')

    plt.xlabel(param_name, fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.grid(True)

    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f"R² Scores Across Values of {param_name}", fontsize=14)

    plt.show()



def plot_dendrogram(linkage_matrix, y_threshold=None, distance="euclidian"):
    sns.set()
    fig = plt.figure(figsize=(11,5))
    # The Dendrogram parameters need to be tuned
        # You can play with 'truncate_mode' and 'p' define what level the dendrogram shows
        # above_threshold_color='k' forces black color for the lines above the threshold)
    dendrogram(linkage_matrix, truncate_mode='level', p=5, color_threshold=y_threshold, above_threshold_color='k')
    if y_threshold:
        plt.hlines(y_threshold, 0, 1000, colors="r", linestyles="dashed")
    plt.title(f'Hierarchical Clustering Dendrogram: Ward Linkage', fontsize=21)
    plt.xlabel('Number of points in node (or index of point if no parenthesis)')
    plt.ylabel(f'{distance.title()} Distance', fontsize=13)
    plt.show()


# def compute_avg_silhouette_scores(df, cluster_factory,range_clusters=range(10)
#                               ):
#     """
#     Compute average silhouette scores for a range of cluster numbers.

#     Returns a pandas DataFrame with columns: ['n_clusters', 'avg_silhouette'].
#     """
#     results = []

#     for nclus in range_clusters:
#         # Skip invalid case
#         if nclus <= 1:
#             continue

#         clust = cluster_factory(n_clusters=nclus)
#         #if hasattr(clust,"fit_predict"):
#         cluster_labels = clust.fit_predict(df)
        
#         silhouette_avg = silhouette_score(df, cluster_labels)
#         results.append({"n_clusters": nclus, "avg_silhouette": silhouette_avg})

#         # Optional: print here if you want
#         print(f"For n_clusters = {nclus}, the average silhouette_score is: {silhouette_avg:.4f}")

#     return pd.DataFrame(results)


def compute_avg_silhouette_scores(
    df,
    param_values,
    cluster_factory,
    param_name="param",
):
    """
    Compute average silhouette scores for a range of parameter values.
    """
    results = []

    for val in param_values:
        clust = cluster_factory(val)

        # fit_predict if available, otherwise fit + labels_
        if hasattr(clust, "fit_predict"):
            labels = clust.fit_predict(df)
        else:
            clust.fit(df)
            labels = clust.labels_

        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            print(f"Skipping {param_name}={val}: only {n_clusters} cluster found.")
            continue

        silhouette_avg = silhouette_score(df, labels)
        results.append(
            {
                param_name: val,
                "n_clusters": n_clusters,
                "avg_silhouette": silhouette_avg,
            }
        )

        print(
            f"For {param_name} = {val}, "
            f"n_clusters = {n_clusters}, "
            f"average silhouette_score = {silhouette_avg:.4f}"
        )

    return pd.DataFrame(results)




def plot_silhouette_results(
    results_df,
    x_col="n_clusters",
    y_col="avg_silhouette",
    hue_param_name=None,
    param_name=None,
    highlight_best=True,
    figsize=(10, 6),
    title="Silhouette Score Across Parameter Values",
    xlabel=None,
    ylabel="Average Silhouette Score",
    marker="o"
):
    """
    Plot silhouette results with optional best-point annotation and labels.

    """

    if param_name is None:
        param_name = x_col  # default label

    if xlabel is None:
        xlabel = param_name.replace("_", " ").title()

    plt.figure(figsize=figsize)

    sns.lineplot(
        data=results_df,
        x=x_col,
        y=y_col,
        hue=hue_param_name,
        marker=marker,

        sort=False
    )

    # Highlight and annotate best parameter
    if highlight_best:
        best_row = results_df.loc[results_df[y_col].idxmax()]
        bx, by = best_row[x_col], best_row[y_col]

        label = f"  {param_name} = {bx}; Score = {by:.2f}"

        plt.scatter(bx, by, color="red", s=100, edgecolor="black", zorder=5)
        plt.text(
            bx, by,
            label,
            fontsize=11,
            ha="left",
            va="bottom",
            color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
        )

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_silhouette_for_k(
    df,
    n_clusters,
    clusterer_factory,
):
    """
    Plot silhouette diagram for a given number of clusters, using any clustering
    algorithm following the sklearn API.

    """
    if n_clusters <= 1:
        raise ValueError("n_clusters must be greater than 1 to compute silhouette.")

    

    # Create and fit clusterer
    clusterer = clusterer_factory(n_clusters)
    if hasattr(clusterer, "fit_predict"):
        cluster_labels = clusterer.fit_predict(df)
    else:
        clusterer.fit(df)
        cluster_labels = clusterer.labels_

    silhouette_avg = silhouette_score(df, cluster_labels)
    sample_silhouette_values = silhouette_samples(df, cluster_labels)

    plt.figure(figsize=(13, 7))

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.title(f"Silhouette plot for {n_clusters} clusters")
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    xmin = np.round(sample_silhouette_values.min() - 0.1, 2)
    xmax = np.round(sample_silhouette_values.max() + 0.1, 2)
    plt.xlim([xmin, xmax])
    plt.xticks(np.arange(xmin, xmax, 0.1))

    plt.ylim([0, len(df) + (n_clusters + 1) * 10])
    plt.yticks([])

    plt.show()

def plot_heatmap_barplot_clusters(df, label):
    sns.set(style="whitegrid")
    label_counts = df[label].value_counts().sort_index()

    fig, axes = plt.subplots(1,2, figsize=(12,5), width_ratios=[.6,.4], tight_layout=True)
    pop_mean = df.mean()
    hc_profile = df.groupby(label).mean().T
    df_concat_pop = pd.concat([hc_profile, 
                           pd.Series(pop_mean, 
                                        index=hc_profile.index, 
                                        name='Population Mean'
                            )
                           ],
                           axis=1)
    
    sns.heatmap(df_concat_pop, ax=axes[0], center=0, cmap='PiYG')

    axes[0].set_xlabel("Cluster Labels")
    axes[0].set_title("Heatmap of Cluster Means")


    sns.barplot(x=label_counts.index, y=label_counts.values, ax=axes[1])
    axes[1].set_title("Cluster Sizes")
    axes[1].set_xlabel("Cluster Labels")

    fig.suptitle("Cluster Profiling:\nHierarchical Clustering with 4 Clusters")
    plt.show()



def plot_k_distance_curve(
    df,
    feature_cols,
    n_neighbors=20,
    eps_line=None,
    figsize=(10, 6),
    title="K-Distance Graph for eps Selection",
    xlabel="Sorted Sample Index",
    ylabel=None,
    show=True,
):
    """
    Computes and plots the K-distance curve used to estimate a suitable eps for DBSCAN.
    """

    if ylabel is None:
        ylabel = f"Distance to {n_neighbors}th Nearest Neighbor"

    # Fit NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(df[feature_cols])

    # Compute k-distances (last column of kneighbors output)
    distances, _ = neigh.kneighbors(df[feature_cols])
    distances = np.sort(distances[:, -1])  # take farthest neighbor, then sort

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(distances)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Optional eps candidate
    if eps_line is not None:
        plt.axhline(y=eps_line, color="red", linestyle="--", label=f"Candidate eps = {eps_line}")
        plt.legend()

    if show:
        plt.show()


def compute_gmm_bic_aic(
    df,
    feature_cols,
    n_components_list,
    covariance_types=("full", "tied", "diag", "spherical"),
    random_state=1,
    n_init=10,
):
    X = df[feature_cols]
    rows = []

    for cov in covariance_types:
        for k in n_components_list:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                n_init=n_init,
                random_state=random_state
            ).fit(X)

            rows.append({
                "n_components": k,
                "covariance_type": cov,
                "bic": gmm.bic(X),
                "aic": gmm.aic(X),
            })

    return pd.DataFrame(rows)



def plot_gmm_ic_scores(
    ic_df,
    x_col="n_components",
    cov_col="covariance_type",
    bic_col="bic",
    aic_col="aic",
    figsize=(12,5),
    bic_title="BIC vs n_components",
    aic_title="AIC vs n_components",
    highlight_best=True
):
    """
    Plots BIC and AIC curves from a Gaussian Mixture model tuning grid.
    """

    plt.figure(figsize=figsize)

    # --- BIC subplot ---
    ax1 = plt.subplot(1,2,1)
    sns.lineplot(
        data=ic_df,
        x=x_col,
        y=bic_col,
        hue=cov_col,
        marker="o",
        ax=ax1
    )
    ax1.set_title(bic_title)
    ax1.grid(True)



    # --- AIC subplot ---
    ax2 = plt.subplot(1,2,2)
    sns.lineplot(
        data=ic_df,
        x=x_col,
        y=aic_col,
        hue=cov_col,
        marker="s",
        ax=ax2
    )
    ax2.set_title(aic_title)
    ax2.grid(True)

   

    plt.tight_layout()
    plt.show()




def evaluate_final_cluster_models(
    df,
    feature_cols,
    models_dict,
    r2_fn
):
    """
    Evaluate final clustering models with silhouette and R².
    """
    X = df[feature_cols]
    rows = []

    for name, clusterer in models_dict.items():
        # fit + labels
        if hasattr(clusterer, "fit_predict"):
            labels = clusterer.fit_predict(X)
        else:
            clusterer.fit(X)
            labels = clusterer.labels_

        # silhouette
        sil = silhouette_score(X, labels)

        # R²
        df_concat = pd.concat(
            [df[feature_cols], pd.Series(labels, name="labels", index=df.index)],
            axis=1
        )
        r2 = r2_fn(df_concat)

        rows.append({
            "model": name,
            "silhouette": sil,
            "r2": r2
        })

        print(f"{name}: silhouette={sil:.4f}, R²={r2:.4f}")

    return pd.DataFrame(rows)



def evaluate_final_cluster_models(
    df,
    feature_cols,
    models_dict,
    r2_fn
):
    """
    Evaluate final clustering models with:
    - Silhouette
    - R²
    - Calinski–Harabasz (CH)
    - Davies–Bouldin Index (DBI)
    """
    X = df[feature_cols]
    rows = []

    for name, clusterer in models_dict.items():
        # fit + labels
        if hasattr(clusterer, "fit_predict"):
            labels = clusterer.fit_predict(X)
        else:
            clusterer.fit(X)
            labels = clusterer.labels_

        n_clusters = len(np.unique(labels))

        if n_clusters < 2:
            print(f"{name}: only {n_clusters} cluster found → metrics set to NaN.")
            sil = np.nan
            ch = np.nan
            dbi = np.nan
        else:
            # silhouette
            sil = silhouette_score(X, labels)

            # Calinski–Harabasz (higher is better)
            ch = calinski_harabasz_score(X, labels)

            # Davies–Bouldin (lower is better)
            dbi = davies_bouldin_score(X, labels)

      
        df_concat = pd.concat(
            [df[feature_cols], pd.Series(labels, name="labels", index=df.index)],
            axis=1
        )
        r2 = r2_fn(df_concat)

        rows.append({
            "model": name,
            "n_clusters": n_clusters,
            "silhouette": sil,
            "r2": r2,
            "calinski_harabasz": ch,
            "davies_bouldin": dbi
        })

        print(
            f"{name}: silhouette={sil:.4f} | "
            f"R²={r2:.4f} | "
            f"CH={ch:.2f} | "
            f"DBI={dbi:.4f}"
            if n_clusters >= 2 else
            f"{name}: R²={r2:.4f} (no valid cluster metrics)"
        )

    return pd.DataFrame(rows)



def plot_parallel_cluster_profiles(
    df,
    label_col="labels",
    figsize=(16,7),
    colormap="tab10",
    linewidth=2,
    rotation=70,
    baseline=True
):
    """
    Plot a parallel coordinates chart for cluster mean profiles.

    """

    # Ensure labels are strings
    
    km_profile = df.groupby(label_col).mean().T
    km_profile = km_profile.T.reset_index()
    df_plot = km_profile.copy()
    df_plot[label_col] = df_plot[label_col].astype(str)

    fig, ax = plt.subplots(figsize=figsize)

    parallel_coordinates(
        df_plot,
        label_col,
        colormap=colormap,
        linewidth=linewidth,
        ax=ax
    )

    # Optional horizontal reference line
    if baseline:
        ax.axhline(0, linestyle="-.", color="black", alpha=0.7)

    ax.set_title("Cluster Profiling with KMeans Clustering:\nParallel Coordinates Plot")
    ax.set_xlabel("Cluster Labels")
    plt.xticks(rotation=rotation)

    plt.tight_layout()
    plt.show()