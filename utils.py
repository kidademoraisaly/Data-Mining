
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd 
import plotly.graph_objects as go
import re

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