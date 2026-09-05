"""Steam games dashboard: catalog explorer + pre-launch owners predictor.

Run locally with:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

import theme

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "games.parquet"
MODEL = ROOT / "models" / "owners_classifier.joblib"

# Only the columns the dashboard actually uses. The full frame is ~188MB in
# memory, mostly from text and list columns nothing here reads; naming columns
# explicitly means pyarrow never loads the rest at all, which matters on
# Streamlit Community Cloud's ~1GB tier.
APP_COLUMNS = [
    "app_id", "name", "price", "price_status", "estimated_owners",
    "genres", "tags", "positive", "negative", "positive_ratio",
    "release_year", "platform_count", "windows", "mac", "linux",
    "genres_count", "tags_count", "has_achievements",
]

st.set_page_config(page_title="Steam Catalog", page_icon="◗", layout="wide")
theme.apply()


@st.cache_data
def load_games():
    return pd.read_parquet(DATA, columns=APP_COLUMNS)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL)


def bucket_bounds(label):
    low, high = label.split(" - ")
    return int(low), int(high)


def short_label(label):
    """'20000 - 50000' -> '20K-50K', for axis labels that have to fit."""
    if label.endswith("+"):
        return f"{int(label[:-1]) // 1_000_000}M+"
    if label == "0 - 0":
        # Not games that launched and sold nothing: these are overwhelmingly
        # unreleased, delisted or placeholder entries, so "0-0" would mislead.
        return "none yet"

    def unit(n):
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        if n >= 1_000:
            return f"{n // 1_000}K"
        return str(n)

    low, high = bucket_bounds(label)
    return f"{unit(low)}-{unit(high)}"


games = load_games()
bundle = load_bundle()

owner_order = sorted(games["estimated_owners"].unique(), key=bucket_bounds)
short_order = [short_label(b) for b in owner_order]
BUCKET_COLOURS = theme.ramp(len(owner_order))


def owners_scale():
    """One colour per owners bucket, shared by every chart on the page."""
    return alt.Scale(domain=short_order, range=BUCKET_COLOURS)


# ------------------------------------------------------------------- masthead
st.markdown('<div class="eyebrow">Steam catalog · 139,580 games</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="lede">Two thirds of Steam never finds <em>20,000 players</em>.</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="standfirst">Every game released on Steam, sized by the audience it '
    "actually reached. Explore the catalog below, or predict how far a game might "
    "get using only what is knowable before it ships.</p>",
    unsafe_allow_html=True,
)

# The signature element: the whole catalog as one proportional strip. Band
# width is share of catalog, so the long tail is not described, it is shown.
# Positions are computed explicitly rather than left to Altair's stacking,
# because the labels need to know exactly where each band starts and ends.
counts_all = games["estimated_owners"].value_counts().reindex(owner_order).dropna()
strip = counts_all.reset_index()
strip.columns = ["bucket", "games"]
strip["label"] = strip["bucket"].map(short_label)
strip["share"] = strip["games"] / strip["games"].sum() * 100
strip["x1"] = strip["share"].cumsum()
strip["x0"] = strip["x1"] - strip["share"]
strip["mid"] = (strip["x0"] + strip["x1"]) / 2

bands = alt.Chart(strip).mark_rect(stroke=theme.PAPER, strokeWidth=1.2).encode(
    x=alt.X("x0:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 100], nice=False)),
    x2="x1:Q",
    color=alt.Color("label:N", scale=owners_scale(), sort=short_order, legend=None),
    tooltip=[
        alt.Tooltip("bucket:N", title="owners"),
        alt.Tooltip("games:Q", title="games", format=","),
        alt.Tooltip("share:Q", title="share of catalog", format=".2f"),
    ],
)

# Only the dominant band is labelled in place. Anything narrower overflows its
# band at phone widths, and the rail below plus tooltips already carry the rest.
# Ink rather than white: the light end of the ramp cannot carry white text.
roomy = strip[strip["share"] > 25].copy()
roomy["caption"] = roomy["label"] + "  " + roomy["share"].round(1).astype(str) + "%"
labels = alt.Chart(roomy).mark_text(
    align="center", baseline="middle", font=theme.MONO, fontSize=11, color=theme.INK,
).encode(x=alt.X("mid:Q", scale=alt.Scale(domain=[0, 100], nice=False)), text="caption:N")

st.altair_chart((bands + labels).properties(height=58), width="stretch")

st.markdown(
    '<div class="stripnote"><span>banded by audience reached · width is share of '
    'catalog · "none yet" is mostly unreleased and delisted listings</span>'
    "<span>4 games ever passed 100M ◗</span></div>",
    unsafe_allow_html=True,
)

st.write("")
explore_tab, predict_tab = st.tabs(["Explore the catalog", "Predict a game's reach"])


# ---------------------------------------------------------------- explore tab
with explore_tab:
    st.sidebar.header("Filters")

    all_genres = sorted({g for row in games["genres"] for g in row})
    chosen_genres = st.sidebar.multiselect("Genres", all_genres, placeholder="Any genre")

    price_lo, price_hi = st.sidebar.slider(
        "Price ($)", 0.0, 60.0, (0.0, 60.0), step=1.0,
        # Dollar signs are escaped throughout: Streamlit renders markdown, and a
        # pair of unescaped $ is parsed as LaTeX math delimiters.
        help="Capped at \\$60; a handful of listings run to \\$107,500 and would flatten the range.",
    )

    years = games["release_year"].dropna()
    year_lo, year_hi = st.sidebar.slider(
        "Release year", int(years.min()), int(years.max()), (2015, int(years.max()))
    )

    view = games[
        games["price"].between(price_lo, price_hi)
        & games["release_year"].between(year_lo, year_hi)
    ]
    if chosen_genres:
        wanted = set(chosen_genres)
        view = view[games["genres"].apply(lambda gs: bool(wanted & set(gs)))]

    if view.empty:
        st.warning("No games match these filters. Widen the price or year range.")
        st.stop()

    st.write("")
    a, b, c, d = st.columns(4)
    a.metric("Games", f"{len(view):,}")
    b.metric("Median price", f"${view['price'].median():.2f}")
    c.metric("Free to play", f"{(view['price_status'] == 'free').mean() * 100:.0f}%")
    rated = view["positive_ratio"].dropna()
    d.metric("Median rating", f"{rated.median() * 100:.0f}%" if len(rated) else "—")

    st.divider()
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="eyebrow">Distribution</div>', unsafe_allow_html=True)
        st.subheader("Where the catalog sits")
        counts = (
            view["estimated_owners"].value_counts().reindex(owner_order).dropna().reset_index()
        )
        counts.columns = ["bucket", "games"]
        counts["label"] = counts["bucket"].map(short_label)
        st.altair_chart(
            alt.Chart(counts).mark_bar(size=15).encode(
                y=alt.Y("label:N", sort=short_order, title=None),
                x=alt.X("games:Q", title="games", scale=alt.Scale(type="symlog")),
                color=alt.Color("label:N", scale=owners_scale(), sort=short_order, legend=None),
                tooltip=[alt.Tooltip("bucket:N", title="owners"),
                         alt.Tooltip("games:Q", format=",")],
            ).properties(height=330),
            width="stretch",
        )
        st.caption("Log scale — on a linear axis every bucket but the first would vanish.")

    with right:
        st.markdown('<div class="eyebrow">Pricing</div>', unsafe_allow_html=True)
        st.subheader("Charm points, not a curve")
        st.altair_chart(
            alt.Chart(view[["price"]]).mark_bar(color=theme.OXBLOOD).encode(
                x=alt.X("price:Q", bin=alt.Bin(maxbins=60), title="price ($)"),
                y=alt.Y("count():Q", title="games"),
            ).properties(height=330),
            width="stretch",
        )
        st.caption(
            "Prices pile up on \\$0.99, \\$4.99 and \\$9.99 rather than spreading "
            "smoothly. Developers anchor to convention."
        )

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="eyebrow">Composition</div>', unsafe_allow_html=True)
        st.subheader("What gets made")
        top = view.explode("genres")["genres"].value_counts().head(12).reset_index()
        top.columns = ["genre", "games"]
        st.altair_chart(
            alt.Chart(top).mark_bar(size=15, color=theme.SLATE).encode(
                x=alt.X("games:Q", title="games"),
                y=alt.Y("genre:N", sort="-x", title=None),
                tooltip=["genre", alt.Tooltip("games:Q", format=",")],
            ).properties(height=350),
            width="stretch",
        )
        st.caption("Indie is two thirds of the whole catalog — Steam is small teams.")

    with right:
        st.markdown('<div class="eyebrow">Price against reach</div>', unsafe_allow_html=True)
        st.subheader("Cheaper does not sell more")
        med = (
            view.groupby("estimated_owners", observed=True)["price"]
            .agg(["median", "size"]).reindex(owner_order).dropna().reset_index()
        )
        med.columns = ["bucket", "median_price", "games"]
        med["label"] = med["bucket"].map(short_label)
        st.altair_chart(
            alt.Chart(med).mark_bar(size=15).encode(
                y=alt.Y("label:N", sort=short_order, title=None),
                x=alt.X("median_price:Q", title="median price ($)"),
                color=alt.Color("label:N", scale=owners_scale(), sort=short_order, legend=None),
                tooltip=[alt.Tooltip("bucket:N", title="owners"),
                         alt.Tooltip("median_price:Q", title="median price", format="$.2f"),
                         alt.Tooltip("games:Q", title="games", format=",")],
            ).properties(height=350),
            width="stretch",
        )
        st.caption(
            "Median price rises with reach. Bands near the bottom rest on very few "
            "games — hover for the count before reading much into them."
        )

    st.write("")
    st.markdown('<div class="eyebrow">Top of the filtered set</div>', unsafe_allow_html=True)
    st.dataframe(
        view.assign(rank=view["estimated_owners"].map({b: i for i, b in enumerate(owner_order)}))
        .sort_values(["rank", "positive"], ascending=[False, False])
        .head(25)[["name", "estimated_owners", "price", "positive", "negative", "release_year"]]
        .rename(columns={
            "name": "Game", "estimated_owners": "Owners", "price": "Price",
            "positive": "Positive", "negative": "Negative", "release_year": "Year",
        }),
        width="stretch", hide_index=True,
    )


# ---------------------------------------------------------------- predict tab
with predict_tab:
    st.markdown('<div class="eyebrow">Pre-launch estimate</div>', unsafe_allow_html=True)
    st.subheader("How far might this game get?")
    st.markdown(
        '<div class="note">Both models see <strong>only pre-launch information</strong>. '
        "Review scores, playtime and concurrent players are deliberately excluded — a "
        "model given those would score well and mean nothing, since none of them exist "
        "on launch day.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("predict"):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            price = st.number_input("Price ($)", 0.0, 200.0, 9.99, step=1.0)
            genres_in = st.multiselect("Genres", bundle["genre_vocab"], default=["Indie"])
            categories_in = st.multiselect(
                "Categories", bundle["category_vocab"], default=["Single-player"]
            )
            tags_in = st.multiselect("Tags", bundle["tag_vocab"], default=["Singleplayer"])

        with c2:
            platforms = st.multiselect(
                "Platforms", ["Windows", "Mac", "Linux"], default=["Windows"]
            )
            achievements = st.checkbox("Has Steam achievements", value=True)
            month = st.selectbox("Release month", list(range(1, 13)), index=9)
            publisher_games = st.number_input(
                "Games previously shipped by this publisher", 0, 5000, 1,
                help="Separates established publishers from first-time developers.",
            )
            description = st.text_area(
                "Store short description", height=120,
                value="Climb a mountain of odds and ends riding a pogo stick.",
                help="Only its length is used as a feature, never its content.",
            )

        submitted = st.form_submit_button("Estimate reach", type="primary")

    if submitted:
        values = {
            "price": price,
            "platform_count": float(len(platforms)),
            "has_achievements": float(achievements),
            "description_length": float(len(description)) if description else None,
            "genres_count": float(len(genres_in)),
            "categories_count": float(len(categories_in)),
            "release_month": float(month),
            "publisher_game_count": float(publisher_games),
        }
        for genre in bundle["genre_vocab"]:
            values[f"genre_{genre}"] = float(genre in genres_in)
        for tag in bundle["tag_vocab"]:
            values[f"tag_{tag}"] = float(tag in tags_in)
        for category in bundle["category_vocab"]:
            values[f"cat_{category}"] = float(category in categories_in)

        def predict_with(model, columns):
            row = pd.DataFrame([{c: values.get(c) for c in columns}], columns=columns)
            probs = pd.Series(model.predict_proba(row)[0], index=model.classes_)
            return model.predict(row)[0], probs.sort_values(ascending=False)

        results = {
            "With tag features": (
                predict_with(bundle["model_with_tags"], bundle["columns_with_tags"]),
                bundle["metrics"]["with_tags"],
            ),
            "Without tag features": (
                predict_with(bundle["model_no_tags"], bundle["columns_no_tags"]),
                bundle["metrics"]["no_tags"],
            ),
        }

        st.divider()
        for column, (name, ((prediction, probs), metrics)) in zip(
            st.columns(2, gap="large"), results.items()
        ):
            with column:
                st.markdown(f'<div class="eyebrow">{name}</div>', unsafe_allow_html=True)
                st.metric("Estimated owners", short_label(prediction),
                          help=f"Full bucket: {prediction}")

                chart_df = probs.head(5).reset_index()
                chart_df.columns = ["bucket", "probability"]
                chart_df["label"] = chart_df["bucket"].map(short_label)
                st.altair_chart(
                    alt.Chart(chart_df).mark_bar(size=15).encode(
                        x=alt.X("probability:Q", title="probability",
                                axis=alt.Axis(format="%")),
                        y=alt.Y("label:N", sort="-x", title=None),
                        color=alt.Color("label:N", scale=owners_scale(),
                                        sort=short_order, legend=None),
                        tooltip=[alt.Tooltip("bucket:N", title="owners"),
                                 alt.Tooltip("probability:Q", format=".1%")],
                    ).properties(height=170),
                    width="stretch",
                )
                st.caption(
                    f"macro F1 {metrics['macro_f1']:.3f} · "
                    f"within one bucket {metrics['within_one']:.1%}"
                )

        st.write("")
        st.markdown(
            '<div class="note"><strong>Why two models, and why the weaker one may be '
            "the honest one.</strong><br><br>The model <em>with</em> tag features scores "
            "better on held-out catalog data. But it learned that relationship from "
            "<em>mature</em> games, whose tag lists were filled in by years of community "
            "tagging — Steam tags come from players, not developers. An unreleased game "
            "carries only the handful of tags its developer picked, which that model "
            "reads as evidence of an unpopular game.<br><br>The model <em>without</em> "
            "tag features scores lower and cannot be fooled that way. When the two "
            "disagree, the gap is usually this mismatch rather than a real disagreement "
            "about the game.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()
    st.caption(
        "Both models reach roughly 0.19–0.23 macro F1 across nine owners buckets — modest. "
        "They rank games far better than they classify them: about 80% of predictions land "
        "within one bucket of the truth, and games that reached millions of owners are "
        "essentially never predicted as flops. Most of what drives a game's reach — "
        "marketing, quality, timing, streamer coverage — is absent from this dataset entirely."
    )
