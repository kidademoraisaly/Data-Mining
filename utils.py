
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd 
<<<<<<< HEAD
import plotly.graph_objects as go

=======
>>>>>>> 37fb2ad (add data_exploration.ipyndb)

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

def pairplot(df, columns):
    sns.pairplot(df[columns], diag_kind="hist")
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Pairwise Relationship of Numerical Variables", fontsize=20)
    plt.show()

# def plot_countplots(df, columns, n_rows=2, title="Categorical Variables' Absolute Counts",  figsize = (20, 11)):
#     sns.set_style("white")

#     # Grid layout
  
#     n_cols = max(1, ceil(len(columns) / max(1, n_rows)))
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
#     axes = np.array(axes).ravel()  # flatten safely
#     # Plot each categorical feature
#     for ax, feat in zip(axes, columns):
#         order = df[feat].value_counts().index  # order by frequency
#         sns.countplot(x=df[feat], ax=ax, order=order)
#         #ax.set_title(str(feat), y=-0.13)
#     # Hide unused axes
#     for ax in axes[len(columns):]:
#         ax.set_visible(False)
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()


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
    return fig, axes




def plot_bivariate_relative(df, cols, figsize=(15, 20), n_rows=1, top_k=10):
    """
    Plot relative stacked bar charts for all combinations of categorical columns.
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

        pd.crosstab(df_copy[x], df_copy[y], normalize='index').plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'{x} vs {y} (Relative counts)')
        axes[i].legend(loc=(1.02, 0))
    
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
