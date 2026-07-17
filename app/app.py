"""
============================================================
 Johor Property Price Advisor  (v4 - dark UI)
 BS23124 Data Mining Mini Project
------------------------------------------------------------
 - Dark "data-mining" theme (see .streamlit/config.toml)
 - Glass KPI cards, all charts in interactive Plotly
 - 3 pages: Price Check / Data Dashboard / Hidden Patterns
 - EVERY chart is wrapped in try/except: if one chart fails
   on a given machine it shows a small note instead of
   crashing the whole app.
 Deps (all already installed): streamlit, pandas, numpy,
   joblib, matplotlib(unused now), plotly, mlxtend.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---- resolve data/model files relative to this script, not the process's
#      current working directory. Streamlit Cloud runs the app with the repo
#      root as the working directory (not the app/ folder), so a bare
#      "house_model.pkl" resolves locally but 404s on Streamlit Cloud with
#      FileNotFoundError. Building an absolute path from __file__ makes this
#      work identically locally and when deployed. ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def _p(filename):
    return os.path.join(BASE_DIR, filename)

# ---- palette (dark / techy) ----
BG, CARD, TEXT, MUTED = "#0D1117", "#161B22", "#E6EDF3", "#8B949E"
ACCENT, BLUE, RED, GREEN = "#00E0A3", "#58A6FF", "#FF6B6B", "#3FB950"
MODEL_ERROR = 0.15

AREA_CENTROIDS = {
    "Johor Bahru": (1.4927, 103.7414), "Skudai": (1.5334, 103.6773),
    "Iskandar Puteri": (1.4239, 103.6318), "Kulai": (1.66, 103.59),
    "Pasir Gudang": (1.469, 103.8988), "Johor Jaya": (1.537, 103.789),
    "Ulu Tiram": (1.598, 103.78), "Kluang": (2.0273, 103.3173),
    "Batu Pahat": (1.8538, 102.9337), "Muar": (2.0442, 102.5689),
    "Segamat": (2.5138, 102.8135), "Pontian": (1.4805, 103.3895),
    "Bukit Indah": (1.513, 103.678), "Mount Austin": (1.5627, 103.7797),
    "Taman Molek": (1.529, 103.776), "Gelang Patah": (1.394, 103.582),
    "Permas Jaya": (1.5034, 103.8186),
}

# ------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(_p("house_model.pkl"))

@st.cache_data
def load_meta():
    meta = json.load(open(_p("app_meta.json"), encoding="utf-8"))
    ref = pd.read_csv(_p("area_reference.csv")).set_index("Area_clean")
    return meta, ref

@st.cache_data
def load_data():
    df = pd.read_csv(_p("johor_final_clean.csv"))
    q1, q2 = df["Price_RM"].quantile([1/3, 2/3])
    return df, float(q1), float(q2)

@st.cache_data
def load_scores():
    try:
        return json.load(open(_p("model_scores.json"), encoding="utf-8"))
    except Exception:
        return {}

model = load_model()
meta, area_ref = load_meta()
df_all, P_LOW, P_HIGH = load_data()
scores = load_scores()

def price_band(p):
    return "Low" if p < P_LOW else ("Mid" if p < P_HIGH else "High")
BAND_LABEL = {"Low": "lower-priced", "Mid": "mid-priced", "High": "higher-priced"}

@st.cache_data
def mine_rules(min_support=0.03, min_conf=0.5):
    d = df_all.copy()
    d["Pb"] = pd.qcut(d["Price_RM"], 3, labels=["Price=Low", "Price=Mid", "Price=High"])
    d["Sb"] = pd.qcut(d["Size_SQFT"], 3, labels=["Size=Small", "Size=Medium", "Size=Large"])
    baskets = [["Area=" + str(r.Area), "Type=" + str(r.Property_Type), str(r.Sb),
                "Beds=" + str(int(round(r.Bedrooms))), str(r.Pb)] for r in d.itertuples()]
    te = TransactionEncoder()
    oh = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
    freq = apriori(oh, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]]

def log_decision(area, ptype, size, beds, baths, asking, point, low, high, verdict):
    try:
        con = sqlite3.connect(_p("johor_property.db"))
        con.execute("""INSERT INTO decision_log
            (query_time,in_area,in_type,in_size,in_beds,in_baths,
             asking_price,predicted_price,range_low,range_high,verdict)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), area, ptype, size, beds, baths,
             asking, round(point), round(low), round(high), verdict))
        con.commit(); con.close()
    except Exception:
        pass

# ------------------------------------------------------------
# Styling helpers
# ------------------------------------------------------------
st.set_page_config(page_title="Johor Property Price Advisor", page_icon="house", layout="wide")
st.markdown(f"""
<style>
    .stApp {{ background:{BG}; }}
    h1,h2,h3 {{ color:{TEXT}; letter-spacing:-0.5px; }}
    .note {{ font-size:13px; color:{MUTED}; line-height:1.55; }}
    .banner {{ background:linear-gradient(120deg,#101826,#161B22);
               border:1px solid rgba(255,255,255,.08); border-radius:18px;
               padding:26px 30px; margin-bottom:14px; }}
    .banner h1 {{ margin:0; font-size:30px; }}
    .glass {{ background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
              border-radius:14px; padding:18px 20px; backdrop-filter:blur(10px);
              box-shadow:0 4px 30px rgba(0,0,0,.15); }}
    .kpi-t {{ color:{MUTED}; font-size:13px; font-weight:500; margin:0; }}
    .kpi-v {{ color:{ACCENT}; font-size:30px; font-weight:700; margin:6px 0 2px 0; }}
    .kpi-s {{ color:{BLUE}; font-size:12px; }}
    .verdict {{ font-size:22px; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

def kpi(col, title, value, sub):
    col.markdown(f"<div class='glass'><p class='kpi-t'>{title}</p>"
                 f"<p class='kpi-v'>{value}</p><span class='kpi-s'>{sub}</span></div>",
                 unsafe_allow_html=True)

def dark(fig, h=330, legend=True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT, height=h,
                      margin=dict(l=10, r=10, t=40, b=10),
                      showlegend=legend)
    return fig

def make_map(dfm, color, size=None, hover=None, zoom=8, size_max=None, title=None):
    """Robust map: prefer new px.scatter_map (MapLibre), fall back to scatter_mapbox."""
    kw = dict(lat="lat", lon="lon", color=color, zoom=zoom,
              color_continuous_scale="Tealgrn")
    if size:     kw["size"] = size
    if size_max: kw["size_max"] = size_max
    if hover:    kw["hover_name"] = hover
    if title:    kw["title"] = title
    try:
        fig = px.scatter_map(dfm, **kw)
        fig.update_layout(map_style="carto-darkmatter")
    except Exception:
        fig = px.scatter_mapbox(dfm, **kw)
        fig.update_layout(mapbox_style="carto-darkmatter")
    return fig

st.markdown(f"<div class='banner'><h1>\U0001F3E0 Johor Property Price Advisor</h1>"
            f"<p class='note' style='margin-top:6px;'>A data-mining decision-support tool that helps "
            f"ordinary buyers judge whether a Johor listing's asking price is fair \u2014 with an estimate, "
            f"comparables, and hidden patterns mined from {len(df_all):,} real listings.</p></div>",
            unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Price Check", "Data Dashboard", "Hidden Patterns"])


# ============================================================
#  PAGE 1: Price Check
# ============================================================
if page == "Price Check":
    left, right = st.columns([1, 1.25])

    with left:
        st.subheader("House details")
        area = st.selectbox("Area", meta["areas"], index=meta["areas"].index("Johor Bahru"))
        ptype = st.selectbox("Property type", meta["types"],
                             index=meta["types"].index("Terrace House")
                             if "Terrace House" in meta["types"] else 0)
        size = st.number_input("Built-up area (sqft)", 200, 20000, 1500, 50)
        cc1, cc2 = st.columns(2)
        beds = cc1.number_input("Bedrooms", 1, 10, 3)
        baths = cc2.number_input("Bathrooms", 1, 10, 2)
        asking = st.number_input("Asking price (RM) \u2014 0 = estimate only", 0, 10_000_000, 0, 10_000)
        clicked = st.button("Estimate", type="primary", use_container_width=True)

    if clicked:
        X = pd.DataFrame([{"Area": area, "Property_Type": ptype,
                           "Size_SQFT": size, "Bedrooms": beds, "Bathrooms": baths}])
        point = float(model.predict(X)[0])
        low, high = point * (1 - MODEL_ERROR), point * (1 + MODEL_ERROR)

        with right:
            st.markdown(f"<div class='glass' style='text-align:center;'>"
                        f"<p class='kpi-t'>Reference estimate</p>"
                        f"<p class='kpi-v' style='font-size:40px;'>RM {point:,.0f}</p>"
                        f"<span class='kpi-s'>Reasonable range RM {low:,.0f} \u2013 RM {high:,.0f}</span></div>",
                        unsafe_allow_html=True)

            verdict = "Estimate only"
            if asking > 0:
                vmin, vmax = point * 0.6, point * 1.4
                if asking > high:
                    diff = (asking - point) / point * 100; color, verdict = RED, "Overpriced"
                    advice = f"~{diff:.0f}% above estimate \u2014 leverage to negotiate."
                elif asking < low:
                    diff = (point - asking) / point * 100; color, verdict = GREEN, "Underpriced"
                    advice = f"~{diff:.0f}% below estimate \u2014 possible deal; check age/tenure."
                else:
                    color, verdict = ACCENT, "Fair price"
                    advice = "Within the reasonable range \u2014 looks fair."
                try:
                    g = go.Figure(go.Indicator(
                        mode="gauge+number", value=asking,
                        number={"prefix": "RM ", "valueformat": ",.0f", "font": {"size": 24}},
                        gauge={"axis": {"range": [vmin, vmax], "tickformat": ",.0f"},
                               "bar": {"color": "rgba(0,0,0,0)"},
                               "steps": [{"range": [vmin, low], "color": "#1e3a2a"},
                                         {"range": [low, high], "color": "#1e3050"},
                                         {"range": [high, vmax], "color": "#4a1f24"}],
                               "threshold": {"line": {"color": color, "width": 5},
                                             "thickness": 0.9, "value": asking}}))
                    st.plotly_chart(dark(g, 230, False), use_container_width=True)
                except Exception:
                    st.info("Gauge unavailable.")
                st.markdown(f"<div class='glass' style='text-align:center;border-color:{color}55;'>"
                            f"<span class='verdict' style='color:{color};'>{verdict}</span>"
                            f"<p class='note' style='margin-top:6px;'>Asking RM {asking:,.0f} &middot; {advice}</p></div>",
                            unsafe_allow_html=True)

        st.divider()
        v1, v2 = st.columns(2)

        # comparable listings + map
        with v1:
            comps = df_all[(df_all["Area"] == area) &
                           (df_all["Size_SQFT"].between(size * 0.8, size * 1.2))].copy()
            st.markdown(f"<p class='note'><b>Comparable houses</b> in {area}, \u00b120% of {size:,} sqft</p>",
                        unsafe_allow_html=True)
            if len(comps) >= 3:
                comps["PSF"] = comps["Price_RM"] / comps["Size_SQFT"]
                comps["d"] = (comps["Size_SQFT"] - size).abs()
                show = comps.sort_values("d").head(8)[
                    ["Property_Type", "Size_SQFT", "Bedrooms", "Price_RM", "PSF"]
                ].rename(columns={"Property_Type": "Type", "Size_SQFT": "Size", "Price_RM": "Price", "PSF": "PSF"})
                st.dataframe(show.style.format({"Size": "{:,.0f}", "Price": "RM {:,.0f}",
                             "PSF": "{:,.0f}", "Bedrooms": "{:.0f}"}),
                             use_container_width=True, hide_index=True, height=230)
                if area in AREA_CENTROIDS:
                    try:
                        la, lo = AREA_CENTROIDS[area]; n = len(comps)
                        rng = np.random.default_rng(0)
                        pts = pd.DataFrame({"lat": la + rng.normal(0, 0.006, n),
                                            "lon": lo + rng.normal(0, 0.006, n),
                                            "Price": comps["Price_RM"].values})
                        fig = make_map(pts, color="Price", zoom=10.5)
                        st.plotly_chart(dark(fig, 260, False), use_container_width=True)
                    except Exception:
                        st.info("Map unavailable on this machine.")
            else:
                st.info("Not enough comparable listings.")

        # distribution + "houses like this"
        with v2:
            area_df = df_all[df_all["Area"] == area]
            if len(area_df) >= 10:
                try:
                    fig = px.histogram(area_df, x="Price_RM", nbins=30,
                                       title=f"Price distribution in {area}")
                    fig.update_traces(marker_color=BLUE)
                    fig.add_vline(point, line_color=ACCENT, line_width=2,
                                  annotation_text="Estimate")
                    if asking > 0:
                        fig.add_vline(asking, line_color=RED, line_width=2, line_dash="dash",
                                      annotation_text="Asking")
                    st.plotly_chart(dark(fig, 260, False), use_container_width=True)
                except Exception:
                    st.info("Distribution chart unavailable.")
            profile = df_all[(df_all["Area"] == area) &
                             (df_all["Size_SQFT"].between(size * 0.7, size * 1.3)) &
                             (df_all["Bedrooms"].between(beds - 1, beds + 1))]
            if len(profile) < 12:
                profile = area_df
            if len(profile) >= 12:
                b = profile["Price_RM"].apply(price_band).value_counts(normalize=True)
                st.markdown(f"<div class='glass'><p class='kpi-t'>Houses like this usually&hellip;</p>"
                            f"<p class='note' style='margin-top:4px;'>Among {len(profile)} houses with similar "
                            f"area, size and rooms, <b style='color:{ACCENT};'>{b.max()*100:.0f}% are "
                            f"{BAND_LABEL[b.idxmax()]}</b>. See <b>Hidden Patterns</b> for full mined rules.</p></div>",
                            unsafe_allow_html=True)

        st.markdown("<p class='note' style='margin-top:14px;'>Note: a reference estimate from area, size, "
                    "type and rooms (~15% average error). It excludes property age, renovation, facing, parking "
                    "and urgency of sale \u2014 use it as a negotiation starting point, not a final answer.</p>",
                    unsafe_allow_html=True)
        log_decision(area, ptype, size, beds, baths, asking, point, low, high, verdict)
    else:
        with right:
            st.markdown("<div class='glass'><p class='note'>Enter a house on the left and press "
                        "<b>Estimate</b>. You'll get a price estimate, a high/fair/low gauge, real "
                        "comparable houses on a map, the area's price distribution, and how houses "
                        "like it are usually priced.</p></div>", unsafe_allow_html=True)


# ============================================================
#  PAGE 2: Data Dashboard
# ============================================================
elif page == "Data Dashboard":
    st.subheader("Dataset overview")
    rf = scores.get("Random Forest", {})
    k = st.columns(4)
    kpi(k[0], "Listings mined", f"{len(df_all):,}", "Mudah + Kaggle + PropertyGuru")
    kpi(k[1], "Best model R\u00b2", f"{rf.get('R2', 0):.2f}", "Random Forest")
    kpi(k[2], "Avg error (MAE)", f"RM {rf.get('MAE', 0):,.0f}", "lower is better")
    kpi(k[3], "Areas covered", f"{df_all['Area'].nunique()}", "across Johor")
    st.divider()

    r1c1, r1c2 = st.columns(2)

    # 1. Avg price by area
    with r1c1:
        try:
            ap = (df_all.groupby("Area")["Price_RM"].mean() / 1000).sort_values().reset_index()
            fig = px.bar(ap, x="Price_RM", y="Area", orientation="h",
                         title="Average price by area (RM '000)",
                         color="Price_RM", color_continuous_scale="Tealgrn")
            st.plotly_chart(dark(fig, 420, False), use_container_width=True)
        except Exception:
            st.info("Chart 1 unavailable.")

    # 2. Size vs Price scatter
    with r1c2:
        try:
            d2 = df_all[(df_all["Size_SQFT"] > 0) & (df_all["Size_SQFT"] < 8000)]
            fig = px.scatter(d2, x="Size_SQFT", y="Price_RM", color="Property_Type",
                             opacity=0.5, title="Built-up area vs price")
            st.plotly_chart(dark(fig, 420), use_container_width=True)
        except Exception:
            st.info("Chart 2 unavailable.")

    r2c1, r2c2 = st.columns(2)

    # 3. Model comparison (all three metrics)
    with r2c1:
        try:
            rows = []
            for m, s in scores.items():
                rows.append({"Model": m, "Metric": "R\u00b2", "Value": s.get("R2", 0)})
            md = pd.DataFrame(rows)
            fig = px.bar(md, x="Model", y="Value", color="Model",
                         title="Model R\u00b2 comparison (higher = better)",
                         color_discrete_sequence=[MUTED, BLUE, ACCENT])
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(dark(fig, 360, False), use_container_width=True)
        except Exception:
            st.info("Chart 3 unavailable.")

    # 4. Property type donut
    with r2c2:
        try:
            tc = df_all["Property_Type"].value_counts().reset_index()
            tc.columns = ["Type", "Count"]
            fig = px.pie(tc, names="Type", values="Count", hole=0.55,
                         title="Property type mix")
            st.plotly_chart(dark(fig, 360), use_container_width=True)
        except Exception:
            st.info("Chart 4 unavailable.")

    r3c1, r3c2 = st.columns(2)

    # 5. Correlation heatmap
    with r3c1:
        try:
            corr = df_all[["Price_RM", "Size_SQFT", "Bedrooms", "Bathrooms"]].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Tealgrn",
                            title="Feature correlation", aspect="auto")
            st.plotly_chart(dark(fig, 360, False), use_container_width=True)
        except Exception:
            st.info("Chart 5 unavailable.")

    # 6. Data source donut (web scraping provenance)
    with r3c2:
        try:
            sc = df_all["Source"].value_counts().reset_index()
            sc.columns = ["Source", "Count"]
            fig = px.pie(sc, names="Source", values="Count", hole=0.55,
                         title="Where the data was scraped from",
                         color_discrete_sequence=[ACCENT, BLUE, RED])
            st.plotly_chart(dark(fig, 360), use_container_width=True)
        except Exception:
            st.info("Chart 6 unavailable.")

    r4c1, r4c2 = st.columns(2)

    # 7. Avg price by bedrooms
    with r4c1:
        try:
            bd = df_all[df_all["Bedrooms"].between(1, 7)]
            bp = (bd.groupby("Bedrooms")["Price_RM"].mean() / 1000).reset_index()
            fig = px.bar(bp, x="Bedrooms", y="Price_RM", title="Average price by bedrooms (RM '000)",
                         color="Price_RM", color_continuous_scale="Tealgrn")
            st.plotly_chart(dark(fig, 340, False), use_container_width=True)
        except Exception:
            st.info("Chart 7 unavailable.")

    # 8. Area map coloured by avg price
    with r4c2:
        try:
            ap = df_all.groupby("Area")["Price_RM"].mean().reset_index()
            ap["lat"] = ap["Area"].map(lambda a: AREA_CENTROIDS.get(a, (None, None))[0])
            ap["lon"] = ap["Area"].map(lambda a: AREA_CENTROIDS.get(a, (None, None))[1])
            ap = ap.dropna(subset=["lat", "lon"])
            fig = make_map(ap, color="Price_RM", size="Price_RM", hover="Area",
                           zoom=7.5, size_max=28, title="Average price by location")
            st.plotly_chart(dark(fig, 340, False), use_container_width=True)
        except Exception:
            st.info("Map unavailable on this machine.")


# ============================================================
#  PAGE 3: Hidden Patterns
# ============================================================
elif page == "Hidden Patterns":
    st.subheader("Hidden Patterns \u2014 Association Rule Mining (Apriori)")
    st.markdown(f"<p class='note'>Single listings can't reveal these patterns \u2014 they only emerge across "
                f"all {len(df_all):,} houses. Each house is a basket of features; <b>Apriori</b> mines rules like "
                f"<i>{{Area, rooms}} &rarr; price band</i>, scored by <b>support</b> (how often it appears), "
                f"<b>confidence</b> (how often it holds) and <b>lift</b> (&gt;1 = stronger than chance).</p>",
                unsafe_allow_html=True)

    try:
        rules = mine_rules()
    except Exception:
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
        st.info("Rule mining unavailable on this machine.")

    if len(rules):
        k = st.columns(3)
        kpi(k[0], "Rules mined", f"{len(rules):,}", "min support 0.03")
        kpi(k[1], "Price-band rules", f"{rules['consequents'].str.contains('Price=').sum()}", "predict price")
        kpi(k[2], "Max lift", f"{rules['lift'].max():.1f}\u00d7", "strongest association")
        st.divider()

        c1, c2 = st.columns(2)
        min_lift = c1.slider("Minimum lift", 1.0, 3.5, 1.5, 0.1)
        only_price = c2.checkbox("Only rules predicting a price band", value=True)
        view = rules[rules["lift"] >= min_lift].copy()
        if only_price:
            view = view[view["consequents"].str.contains("Price=")]
        view = view.sort_values("lift", ascending=False)

        g1, g2 = st.columns([1.2, 1])
        with g1:
            st.dataframe(view.head(25).rename(columns={
                "antecedents": "If", "consequents": "Then",
                "support": "Support", "confidence": "Confidence", "lift": "Lift"})
                .style.format({"Support": "{:.2f}", "Confidence": "{:.2f}", "Lift": "{:.2f}"}),
                use_container_width=True, hide_index=True, height=420)
        with g2:
            try:
                fig = px.scatter(view, x="support", y="confidence", size="lift", color="lift",
                                 color_continuous_scale="Tealgrn", title="Rules: support vs confidence",
                                 hover_data={"antecedents": True})
                st.plotly_chart(dark(fig, 420, False), use_container_width=True)
            except Exception:
                st.info("Rule chart unavailable.")

        if len(view):
            t = view.iloc[0]
            st.markdown(f"<div class='glass'><p class='kpi-t'>Strongest rule</p>"
                        f"<p class='note' style='margin-top:4px;'>If a house is "
                        f"<b style='color:{ACCENT};'>{{{t['antecedents']}}}</b>, it tends to be "
                        f"<b style='color:{BLUE};'>{{{t['consequents']}}}</b> "
                        f"(confidence {t['confidence']:.0%}, lift {t['lift']:.1f}\u00d7) \u2014 exactly the "
                        f"kind of hidden pattern a buyer can't read off a single listing.</p></div>",
                        unsafe_allow_html=True)