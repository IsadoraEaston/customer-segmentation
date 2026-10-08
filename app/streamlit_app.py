"""
Customer Segmentation — interactive demo
RFM analysis + K-means on the Online Retail II dataset (UCI).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

DATA_FILE = Path(__file__).parent / "rfm_segments.csv"
FEATURES = ["Recency", "Frequency", "Monetary"]
RANDOM_STATE = 42  # same value as notebook 03, so the segments match

SEGMENTS = {
    "Champions": {
        "color": "#2E7D32",
        "profile": "Frequent, recent and high-spending.",
        "action": "Protect them: loyalty perks, early access, priority service.",
    },
    "At risk": {
        "color": "#E65100",
        "profile": "Valuable customers who haven't returned in months.",
        "action": "Win them back: a personal message or a targeted offer — before they leave for good.",
    },
    "Promising": {
        "color": "#1565C0",
        "profile": "Active recently, but only a few orders so far.",
        "action": "Build the habit: follow-up offers that encourage the next purchase.",
    },
    "Lost": {
        "color": "#757575",
        "profile": "A single purchase, more than a year ago.",
        "action": "Low-cost reactivation at most (e.g. a seasonal email).",
    },
}
ORDER = list(SEGMENTS)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE, index_col="Customer ID")


@st.cache_resource
def fit_model(rfm):
    """Re-trains the model exactly like notebook 03 (log → standardize → K-means)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(np.log1p(rfm[FEATURES]))
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit(X)

    # Map each K-means cluster to the segment name used in the notebook
    cluster_to_segment = (
        pd.Series(rfm["Segment"].values).groupby(kmeans.labels_)
        .agg(lambda s: s.mode()[0]).to_dict()
    )
    return scaler, kmeans, cluster_to_segment


rfm = load_data()
scaler, kmeans, cluster_to_segment = fit_model(rfm)

# ---------- Header ----------
st.title("🛍️ Customer Segmentation")
st.markdown(
    "Who are a retailer's best customers — and who is it about to lose? "
    "This demo segments **5,800+ customers** of a real UK online retailer "
    "using **RFM analysis** and **K-means clustering**."
)

summary = rfm.groupby("Segment").agg(
    Customers=("Recency", "size"),
    Revenue=("Monetary", "sum"),
    Recency=("Recency", "median"),
    Frequency=("Frequency", "median"),
    Monetary=("Monetary", "median"),
).reindex(ORDER)
summary["Share of customers"] = summary["Customers"] / summary["Customers"].sum()
summary["Share of revenue"] = summary["Revenue"] / summary["Revenue"].sum()

champ = summary.loc["Champions"]
c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{len(rfm):,}")
c2.metric("Champions — share of customers", f"{champ['Share of customers']:.0%}")
c3.metric("Champions — share of revenue", f"{champ['Share of revenue']:.0%}")

tab1, tab2, tab3 = st.tabs(["📊 Segments", "🔍 Explore", "🧮 Classify a customer"])

# ---------- Tab 1: segment overview ----------
with tab1:
    cols = st.columns(4)
    for col, name in zip(cols, ORDER):
        row, info = summary.loc[name], SEGMENTS[name]
        with col:
            st.markdown(
                f"<h3 style='color:{info['color']};margin-bottom:0'>{name}</h3>",
                unsafe_allow_html=True,
            )
            st.caption(info["profile"])
            st.markdown(
                f"**{row['Share of customers']:.0%}** of customers  \n"
                f"**{row['Share of revenue']:.0%}** of revenue  \n"
                f"Last purchase: **{row['Recency']:.0f} days** ago  \n"
                f"Orders: **{row['Frequency']:.0f}**  \n"
                f"Typical spend: **£{row['Monetary']:,.0f}**"
            )
            st.info(info["action"])
    st.caption("Typical values are medians.")

    share = summary[["Share of customers", "Share of revenue"]].reset_index().melt(
        id_vars="Segment", var_name="Measure", value_name="Share")
    fig = px.bar(share, x="Segment", y="Share", color="Measure", barmode="group",
                 title="Share of customers vs. share of revenue",
                 color_discrete_sequence=["#90A4AE", "#37474F"])
    fig.update_layout(yaxis_tickformat=".0%", legend_title=None)
    st.plotly_chart(fig, width="stretch")

# ---------- Tab 2: 3D exploration ----------
with tab2:
    st.markdown("Each dot is a customer. Drag to rotate, scroll to zoom, "
                "click a segment in the legend to hide it.")
    plot_df = np.log1p(rfm[FEATURES]).join(rfm["Segment"])
    plot_df.columns = ["log(Recency)", "log(Frequency)", "log(Monetary)", "Segment"]
    fig3d = px.scatter_3d(
        plot_df.reset_index(), x="log(Recency)", y="log(Frequency)", z="log(Monetary)",
        color="Segment", category_orders={"Segment": ORDER},
        color_discrete_map={k: v["color"] for k, v in SEGMENTS.items()},
        hover_name="Customer ID", opacity=0.6, height=650,
    )
    fig3d.update_traces(marker_size=2.5)
    st.plotly_chart(fig3d, width="stretch")
    st.caption("Axes use a log scale: without it, a few very large wholesale "
               "customers would squeeze everyone else into a corner.")

# ---------- Tab 3: classify a new customer ----------
with tab3:
    st.markdown("Enter a customer's purchase history to see which segment the model assigns.")
    a, b, c = st.columns(3)
    recency = a.number_input("Days since last purchase", 1, 800, 30)
    frequency = b.number_input("Number of orders", 1, 500, 3)
    monetary = c.number_input("Total spent (£)", 1.0, 1_000_000.0, 800.0, step=50.0)

    x = scaler.transform(np.log1p(pd.DataFrame([[recency, frequency, monetary]],
                                               columns=FEATURES)))
    segment = cluster_to_segment[int(kmeans.predict(x)[0])]
    info = SEGMENTS[segment]

    st.markdown(
        f"<h2>Segment: <span style='color:{info['color']}'>{segment}</span></h2>",
        unsafe_allow_html=True,
    )
    st.write(info["profile"])
    st.success(f"**Suggested action:** {info['action']}")

# ---------- Footer ----------
st.divider()
st.caption(
    "Data: Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository, "
    "CC BY 4.0. · Code: [GitHub](https://github.com/IsadoraEaston/customer-segmentation) "
    "· Built by [Isabelle D. Easton](https://isadoraeaston.github.io)"
)
