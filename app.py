"""
CDC Provisional Natality 2025 Dashboard
----------------------------------------
Streamlit dashboard for exploring state, monthly, and sex-based
differences in 2025 provisional U.S. birth counts (CDC data).
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & global chart style (kept minimal, per UX requirements)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CDC Provisional Natality 2025 Dashboard",
    page_icon="\U0001F476",
    layout="wide",
)
px.defaults.template = "plotly_white"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_FILENAME = "Provisional_Natality_2025_CDC1.csv"

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Colorblind-accessible palette (Okabe-Ito)
COLOR_FEMALE = "#E69F00"
COLOR_MALE = "#0072B2"
SEQUENTIAL_SCALE = "Blues"

# State-to-abbreviation mapping, built manually since the CSV has no
# FIPS/abbreviation column. Covers all 50 states + DC (no territories
# are present in this dataset). Needed for the choropleth map.
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

EXPECTED_COLUMNS = {
    "state_of_residence", "month", "month_code", "year_code",
    "sex_of_infant", "births",
}


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and validate the provisional natality CSV.

    The CSV lives next to this script so the same relative path works
    both locally and on Streamlit Community Cloud.
    """
    csv_path = Path(__file__).parent / DATA_FILENAME
    df = pd.read_csv(csv_path, encoding="utf-8-sig")  # utf-8-sig strips a BOM if present

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected column(s): {sorted(missing_cols)}")

    if (df["births"] < 0).any():
        raise ValueError("Found negative birth counts in the data.")

    unexpected_sex = set(df["sex_of_infant"].unique()) - {"Female", "Male"}
    if unexpected_sex:
        raise ValueError(f"Unexpected sex_of_infant value(s): {unexpected_sex}")

    unexpected_months = set(df["month"].unique()) - set(MONTH_ORDER)
    if unexpected_months:
        raise ValueError(f"Unexpected month value(s): {unexpected_months}")

    unmapped_states = set(df["state_of_residence"].unique()) - set(STATE_ABBR)
    if unmapped_states:
        raise ValueError(f"State(s) missing from abbreviation map: {unmapped_states}")

    # Preserve chronological month ordering everywhere downstream.
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    df["state_abbr"] = df["state_of_residence"].map(STATE_ABBR)

    return df


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
def init_filter_state(all_states, all_months):
    """Seed session_state with default (all-selected) filters on first run."""
    st.session_state.setdefault("selected_states", list(all_states))
    st.session_state.setdefault("selected_months", list(all_months))
    st.session_state.setdefault("selected_sexes", ["Female", "Male"])


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered dataframe."""
    all_states = sorted(df["state_of_residence"].unique())
    all_months = MONTH_ORDER

    init_filter_state(all_states, all_months)

    st.sidebar.header("Filters")

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Select All", use_container_width=True):
        st.session_state.selected_states = list(all_states)
        st.session_state.selected_months = list(all_months)
        st.session_state.selected_sexes = ["Female", "Male"]
    if col_b.button("Reset Filters", use_container_width=True):
        st.session_state.selected_states = list(all_states)
        st.session_state.selected_months = list(all_months)
        st.session_state.selected_sexes = ["Female", "Male"]

    st.sidebar.multiselect("State / Geography", options=all_states, key="selected_states")
    st.sidebar.multiselect("Month", options=all_months, key="selected_months")
    st.sidebar.multiselect("Infant Sex", options=["Female", "Male"], key="selected_sexes")

    n_states = len(st.session_state.selected_states)
    n_months = len(st.session_state.selected_months)
    n_sexes = len(st.session_state.selected_sexes)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**Active filters:** {n_states} of {len(all_states)} states \u00b7 "
        f"{n_months} of {len(all_months)} months \u00b7 "
        f"{n_sexes} of 2 sexes"
    )

    return df[
        df["state_of_residence"].isin(st.session_state.selected_states)
        & df["month"].isin(st.session_state.selected_months)
        & df["sex_of_infant"].isin(st.session_state.selected_sexes)
    ]


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
def render_kpis(filtered: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)

    total_births = int(filtered["births"].sum())
    n_geo = filtered["state_of_residence"].nunique()

    by_month = filtered.groupby("month", observed=True)["births"].sum()
    avg_per_month = by_month.mean() if len(by_month) else 0

    by_state = filtered.groupby("state_of_residence")["births"].sum()
    top_state = by_state.idxmax() if len(by_state) else "\u2014"
    top_month = by_month.idxmax() if len(by_month) else "\u2014"

    col1.metric("Total Births", f"{total_births:,}")
    col2.metric("Geographies Selected", f"{n_geo:,}")
    col3.metric("Avg Births / Month", f"{avg_per_month:,.0f}")
    col4.metric("Top Geography", str(top_state))
    col5.metric("Top Month", str(top_month))


# ---------------------------------------------------------------------------
# Chart builders (each returns a Plotly figure)
# ---------------------------------------------------------------------------
def chart_monthly_trend(filtered: pd.DataFrame):
    by_month = (
        filtered.groupby("month", observed=True)["births"]
        .sum()
        .reindex(MONTH_ORDER)
        .dropna()
        .reset_index()
    )
    fig = px.line(
        by_month, x="month", y="births", markers=True,
        title="Monthly Birth Trend",
        labels={"month": "Month", "births": "Births"},
    )
    fig.update_traces(hovertemplate="%{x}: %{y:,} births<extra></extra>")
    fig.update_yaxes(rangemode="tozero", tickformat=",")
    return fig


def chart_sex_comparison(filtered: pd.DataFrame):
    by_month_sex = (
        filtered.groupby(["month", "sex_of_infant"], observed=True)["births"]
        .sum()
        .reset_index()
    )
    fig = px.bar(
        by_month_sex, x="month", y="births", color="sex_of_infant",
        barmode="group", category_orders={"month": MONTH_ORDER},
        title="Births by Month and Infant Sex",
        labels={"month": "Month", "births": "Births", "sex_of_infant": "Sex"},
        color_discrete_map={"Female": COLOR_FEMALE, "Male": COLOR_MALE},
    )
    fig.update_yaxes(rangemode="tozero", tickformat=",")
    return fig


def chart_state_ranking(filtered: pd.DataFrame):
    by_state = (
        filtered.groupby("state_of_residence")["births"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig = px.bar(
        by_state, x="births", y="state_of_residence", orientation="h",
        title="Total Births by State (Selected Filters)",
        labels={"births": "Births", "state_of_residence": "State"},
    )
    fig.update_xaxes(rangemode="tozero", tickformat=",")
    fig.update_layout(height=max(400, 18 * len(by_state)))
    return fig


def chart_choropleth(filtered: pd.DataFrame):
    by_state = (
        filtered.groupby(["state_of_residence", "state_abbr"])["births"]
        .sum()
        .reset_index()
    )
    fig = px.choropleth(
        by_state, locations="state_abbr", locationmode="USA-states",
        color="births", scope="usa", color_continuous_scale=SEQUENTIAL_SCALE,
        hover_name="state_of_residence",
        labels={"births": "Births"},
        title="Births by State",
    )
    fig.update_traces(hovertemplate="%{hovertext}: %{z:,} births<extra></extra>")
    return fig


def chart_heatmap(filtered: pd.DataFrame):
    pivot = (
        filtered.groupby(["state_of_residence", "month"], observed=True)["births"]
        .sum()
        .unstack("month")
        .reindex(columns=MONTH_ORDER)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(
        pivot, color_continuous_scale=SEQUENTIAL_SCALE, aspect="auto",
        labels=dict(x="Month", y="State", color="Births"),
        title="State-by-Month Birth Heatmap (Selected States)",
    )
    fig.update_layout(height=max(400, 18 * len(pivot)))
    return fig


def chart_top_bottom(filtered: pd.DataFrame, n: int = 5):
    by_state = filtered.groupby("state_of_residence")["births"].sum().sort_values(ascending=False)
    n = min(n, len(by_state))
    if n == 0:
        return None

    top = by_state.head(n)
    bottom = by_state.tail(n)
    combined = pd.concat([top, bottom]).reset_index()
    combined.columns = ["state_of_residence", "births"]
    combined["group"] = ["Top"] * len(top) + ["Bottom"] * len(bottom)

    fig = px.bar(
        combined, x="births", y="state_of_residence", color="group",
        orientation="h",
        title=f"Top {n} vs Bottom {n} Geographies by Births",
        labels={"births": "Births", "state_of_residence": "State", "group": "Group"},
        color_discrete_map={"Top": COLOR_MALE, "Bottom": COLOR_FEMALE},
    )
    fig.update_xaxes(rangemode="tozero", tickformat=",")
    return fig


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def render_overview_tab(filtered: pd.DataFrame) -> None:
    st.caption("A high-level look at how total births moved across 2025.")
    st.plotly_chart(chart_monthly_trend(filtered), use_container_width=True)


def render_geographic_tab(filtered: pd.DataFrame) -> None:
    st.plotly_chart(chart_choropleth(filtered), use_container_width=True)
    st.plotly_chart(chart_state_ranking(filtered), use_container_width=True)
    fig = chart_top_bottom(filtered)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def render_monthly_sex_tab(filtered: pd.DataFrame) -> None:
    st.plotly_chart(chart_sex_comparison(filtered), use_container_width=True)
    st.plotly_chart(chart_heatmap(filtered), use_container_width=True)


def render_data_table_tab(filtered: pd.DataFrame) -> None:
    search = st.text_input("Search by state name")
    table = filtered.copy()
    if search:
        table = table[table["state_of_residence"].str.contains(search, case=False, na=False)]

    display_cols = ["state_of_residence", "month", "sex_of_infant", "births"]
    st.dataframe(
        table[display_cols].sort_values(["state_of_residence", "month"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "births": st.column_config.NumberColumn("Births", format="%,d"),
            "state_of_residence": "State",
            "month": "Month",
            "sex_of_infant": "Sex",
        },
    )

    csv_bytes = table[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_bytes,
        file_name="filtered_natality_data.csv",
        mime="text/csv",
    )


def render_about_tab() -> None:
    st.markdown(
        """
        ### About the Data

        **Source:** CDC provisional natality data for 2025 (National Center
        for Health Statistics).

        **What's included:** Monthly birth counts by state of residence and
        infant sex for calendar year 2025, covering the 50 states and the
        District of Columbia.

        **Important notes:**
        - These figures are **provisional** and subject to revision as more
          complete data become available.
        - These are **birth counts**, not birth rates \u2014 they are not
          adjusted for population size, so larger states will naturally show
          higher counts.
        """
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.title("\U0001F476 CDC Provisional Natality Dashboard \u2014 2025")
    st.markdown(
        "Explore how U.S. birth counts in 2025 vary by state, month, and "
        "infant sex, using CDC provisional natality data."
    )
    st.warning(
        "**Provisional data:** these figures are preliminary and subject to revision.",
        icon="\u26a0\ufe0f",
    )
    st.info(
        "**Birth counts, not birth rates:** values shown are raw counts and "
        "are not adjusted for population size.",
        icon="\u2139\ufe0f",
    )
    st.caption(
        "Source: Centers for Disease Control and Prevention (CDC), "
        "National Center for Health Statistics."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        df = load_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()

    render_header()
    filtered = render_sidebar(df)

    if filtered.empty:
        st.error(
            "No data matches the current filter selection. "
            "Try adjusting or resetting the filters."
        )
        st.stop()

    render_kpis(filtered)
    st.markdown("---")

    tab_overview, tab_geo, tab_monthly_sex, tab_table, tab_about = st.tabs(
        [
            "Overview",
            "Geographic Analysis",
            "Monthly and Sex Analysis",
            "Data Table and Download",
            "About the Data",
        ]
    )

    with tab_overview:
        render_overview_tab(filtered)
    with tab_geo:
        render_geographic_tab(filtered)
    with tab_monthly_sex:
        render_monthly_sex_tab(filtered)
    with tab_table:
        render_data_table_tab(filtered)
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
