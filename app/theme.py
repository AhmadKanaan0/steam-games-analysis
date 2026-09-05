"""Visual system for the dashboard.

Design direction — "Catalog". The subject of this dataset is a power law:
139,580 releases, two thirds of which never pass 20,000 owners. So colour is
never decorative here, it encodes magnitude: every chart draws from one ramp
running from the cool grey of the long tail to the oxblood of the few games
that reach millions. Type pairs a characterful grotesque for display against a
workmanlike body face and a mono used for every numeral, because tabular
figures are genuinely useful when the whole page is counts and percentages.
"""

import altair as alt
import streamlit as st

INK = "#14161B"
PAPER = "#F7F7F5"
SLATE = "#5B6472"
RULE = "#D9DAD5"
OXBLOOD = "#8B1E3F"

# Magnitude ramp: the long tail is cool and recessive, the hits run hot. The
# cool end still needs weight on paper — the tail is two thirds of every chart,
# and too light a grey reads as an empty bar rather than as the bulk of Steam.
RAMP_ANCHORS = ["#A9AEB6", "#7C8798", "#A65E71", OXBLOOD]

DISPLAY = "Bricolage Grotesque"
BODY = "Public Sans"
MONO = "IBM Plex Mono"


def _lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _hex_to_rgb(value):
    return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))


def ramp(n):
    """n colours interpolated across the magnitude anchors."""
    stops = [_hex_to_rgb(c) for c in RAMP_ANCHORS]
    if n == 1:
        return [RAMP_ANCHORS[0]]

    out = []
    for i in range(n):
        pos = i / (n - 1) * (len(stops) - 1)
        low = min(int(pos), len(stops) - 2)
        r, g, b = _lerp(stops[low], stops[low + 1], pos - low)
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


@alt.theme.register("catalog", enable=True)
def _catalog_theme():
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": "transparent",
                "font": BODY,
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelFont": MONO,
                    "labelFontSize": 10,
                    "labelColor": SLATE,
                    "titleFont": MONO,
                    "titleFontSize": 10,
                    "titleColor": SLATE,
                    "titleFontWeight": "normal",
                    "domainColor": RULE,
                    "tickColor": RULE,
                    "gridColor": "#ECECE8",
                    "gridWidth": 1,
                    "labelPadding": 6,
                },
                "legend": {
                    "labelFont": MONO,
                    "labelFontSize": 10,
                    "titleFont": MONO,
                    "titleFontSize": 10,
                    "labelColor": SLATE,
                    "titleColor": SLATE,
                },
                "bar": {"color": OXBLOOD},
                "range": {"heatmap": ramp(9), "ramp": ramp(9)},
            }
        }
    )


FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,700;12..96,800&"
    "family=Public+Sans:wght@400;500;600&"
    'family=IBM+Plex+Mono:wght@400;500&display=swap">'
)

CSS = f"""
<style>

/* Streamlit sets font-family via its own emotion classes, which outrank plain
   selectors — every font rule here needs !important to land. */
html, body, [class*="st-"], .stMarkdown, p, li, label, .stTabs, button, input, textarea {{
    font-family: '{BODY}', system-ui, sans-serif !important;
}}

/* Streamlit's own chrome is noise on a portfolio page. */
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {{ display: none; }}
[data-testid="stAppViewContainer"] {{ background: {PAPER}; }}
.block-container {{ padding-top: 2.6rem; max-width: 1180px; }}

/* Repeated class selectors: Streamlit's heading rules also use !important, so
   these need to win on specificity rather than on the flag alone. */
h1, h2, h3, [data-testid="stHeading"] h1, [data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {{
    font-family: '{DISPLAY}', system-ui, sans-serif !important;
    letter-spacing: -0.02em;
    color: {INK};
}}
h1 {{ font-weight: 800; }}
h2, h3 {{ font-weight: 700; }}
h2 {{ font-size: 1.32rem !important; }}
h3 {{ font-size: 1.1rem !important; }}

/* Eyebrow: small mono label that sits above a block and names it. */
.eyebrow {{
    font-family: '{MONO}', monospace !important;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {SLATE};
    margin-bottom: 0.5rem;
}}

/* Streamlit wraps heading text in an inner span, so the family has to reach
   that too or it inherits the theme font back. */
.lede.lede.lede, .lede.lede.lede span {{
    font-family: '{DISPLAY}', system-ui, sans-serif !important;
    font-weight: 700;
    font-size: clamp(1.9rem, 4.2vw, 3.1rem);
    line-height: 1.03;
    letter-spacing: -0.035em;
    color: {INK};
}}
.lede.lede.lede {{ margin: 0 0 0.7rem 0; max-width: 20ch; }}
.lede em, .lede em span {{ font-style: normal; color: {OXBLOOD} !important; }}

/* Streamlit adds an anchor link on hover for headings; unwanted here. */
.lede a, [data-testid="stHeadingWithActionElements"] a {{ display: none !important; }}

.standfirst {{
    font-size: 0.98rem;
    line-height: 1.55;
    color: {SLATE};
    max-width: 62ch;
    margin-bottom: 1.6rem;
}}

/* Numerals are mono everywhere: metrics, tables, axis labels. */
[data-testid="stMetricValue"] {{
    font-family: '{MONO}', monospace !important;
    font-size: 1.55rem;
    font-weight: 500;
    color: {INK};
}}
[data-testid="stMetricLabel"] p {{
    font-family: '{MONO}', monospace !important;
    font-size: 0.66rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SLATE};
}}

/* Caption rail under the catalog strip: reading direction left to right, tail
   to hits, matching the strip above it. */
.stripnote {{
    display: flex;
    justify-content: space-between;
    gap: 2rem;
    font-family: '{MONO}', monospace !important;
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {SLATE};
    margin-top: -4px;
}}

hr {{ border-color: {RULE}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.8rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: '{MONO}', monospace !important;
    font-size: 0.76rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0 0 0.6rem 0;
}}
.stTabs [aria-selected="true"] {{ color: {OXBLOOD}; }}

[data-testid="stSidebar"] {{ background: #EFEFEC; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] h2 {{ font-size: 0.7rem; font-family: '{MONO}', monospace !important;
    letter-spacing: 0.14em; text-transform: uppercase; color: {SLATE}; font-weight: 400; }}

.stButton button, .stFormSubmitButton button {{
    font-family: '{MONO}', monospace !important;
    font-size: 0.76rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 2px;
    border: 1px solid {OXBLOOD};
}}

[data-testid="stCaptionContainer"] p {{
    font-size: 0.79rem;
    color: {SLATE};
    line-height: 1.5;
}}

/* Note block: quieter than st.info's default blue wash. */
.note {{
    border-left: 2px solid {OXBLOOD};
    padding: 0.1rem 0 0.1rem 1rem;
    color: {SLATE};
    font-size: 0.88rem;
    line-height: 1.6;
}}
.note strong {{ color: {INK}; }}
</style>
"""


def apply():
    # Two separate calls on purpose: when a <style> block does not lead the
    # string, Streamlit's markdown sanitiser drops it and keeps only the links.
    st.markdown(FONTS, unsafe_allow_html=True)
    st.markdown(CSS, unsafe_allow_html=True)
