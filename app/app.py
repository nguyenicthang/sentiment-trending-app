"""
Streamlit Dashboard — Gộp hoàn chỉnh (Dashboard Charts + AI Deep Learning)
Chạy: streamlit run app/app.py
Yêu cầu: chạy xong preprocessing.py -> features.py -> train_model.py trước
"""
import sys
import os
import re
import io
import base64
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
from wordcloud import WordCloud
from transformers import pipeline

# ── PATH SETUP ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    from trending import (
        compute_trending_score, top_hashtags,
        best_hour, best_platform, sentiment_over_time,
    )
except ImportError:
    st.error("❌ Không tìm thấy trending.py trong thư mục src/. Kiểm tra lại cấu trúc thư mục.")
    st.stop()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── STYLE ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-box {
    background: linear-gradient(135deg,#1B2A4A,#0D7377);
    color:white; padding:1rem 0.8rem; border-radius:10px;
    text-align:center; margin-bottom:.6rem;
  }
  .kpi-val  { font-size:1.8rem; font-weight:700; line-height:1.1; }
  .kpi-lbl  { font-size:.78rem; opacity:.85; margin-top:.25rem; }
  .sec-hdr  {
    border-left:4px solid #0D7377; padding-left:.7rem;
    font-size:1rem; font-weight:600; color:#1B2A4A;
    margin:1.2rem 0 .6rem;
  }
  [data-testid="stSidebarContent"] { background:#1B2A4A; }
  [data-testid="stSidebarContent"] * { color:white !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CLR = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#f59e0b"}
REQUIRED = ["Text", "Sentiment", "Platform", "Likes", "Retweets", "Hashtags", "Year", "Month", "Hour"]
LABEL_MAP_INV = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

# Khởi tạo LABEL_MAP (Dùng để chuẩn hóa dữ liệu đầu vào)
LABEL_MAP = {}
positive_words = ["Positive", "Happy", "Happiness", "Joy", "Love", "Amusement", "Enjoyment", "Admiration", "Affection", "Awe", "Surprise", "Adoration", "Anticipation", "Calmness", "Excitement", "Kind", "Pride", "Elation", "Euphoria", "Contentment", "Serenity", "Gratitude", "Hope", "Empowerment", "Compassion", "Tenderness", "Enthusiasm", "Fulfillment", "Reverence", "Zest", "Hopeful", "Proud", "Grateful", "Empathetic", "Compassionate", "Playful", "Free-spirited", "Inspired", "Confident", "Thrill", "Overjoyed", "Inspiration", "Motivation", "Satisfaction", "Blessed", "Appreciation", "Confidence", "Accomplishment", "Wonderment", "Optimism", "Enchantment", "PlayfulJoy", "Mindfulness", "DreamChaser", "Harmony", "Creativity", "Radiance", "Wonder", "Rejuvenation", "Coziness", "Adventure", "Melodic", "FestiveJoy", "InnerJourney", "Freedom", "Dazzle", "Adrenaline", "ArtisticBurst", "CulinaryOdyssey", "Resilience", "Spark", "Marvel", "Positivity", "Kindness", "Friendship", "Success", "Exploration", "Amazement", "Romance", "Captivation", "Tranquility", "Grandeur", "Emotion", "Energy", "Celebration", "Charm", "Ecstasy", "Colorful", "Hypnotic", "Connection", "Iconic", "Journey", "Engagement", "Touched", "Triumph", "Heartwarming", "Solace", "Breakthrough", "Joy in Baking", "Envisioning History", "Imagination", "Vibrancy", "Mesmerizing", "Culinary Adventure", "Winter Magic", "Thrilling Journey", "Nature's Beauty", "Celestial Wonder", "Creative Inspiration", "Runway Creativity", "Ocean's Freedom", "Whispers of the Past", "Relief", "JoyfulReunion", "Elegance", "Whimsy", "Renewed Effort"]
negative_words = ["Negative", "Anger", "Fear", "Sadness", "Disgust", "Disappointed", "Bitter", "Shame", "Despair", "Grief", "Loneliness", "Jealousy", "Resentment", "Frustration", "Boredom", "Anxiety", "Intimidation", "Helplessness", "Envy", "Regret", "Melancholy", "Bitterness", "Yearning", "Fearful", "Apprehensive", "Overwhelmed", "Jealous", "Devastated", "Frustrated", "Envious", "Dismissive", "Heartbreak", "Betrayal", "Suffering", "EmotionalStorm", "Isolation", "Disappointment", "LostLove", "Exhaustion", "Sorrow", "Darkness", "Desperation", "Ruins", "Desolation", "Loss", "Heartache", "Obstacle", "Pressure", "Miscalculation", "Challenge", "Embarrassed", "Sad", "Hate", "Bad", "Angry", "Solitude"]
neutral_words = ["Neutral", "Confusion", "Curiosity", "Indifference", "Numbness", "Nostalgia", "Ambivalence", "Determination", "Arousal", "Acceptance", "Contemplation", "Reflection", "Intrigue", "Pensive", "Immersion", "Suspense", "Excited", "Bittersweet", "Sympathy", "Mischievous"]

for w in positive_words: LABEL_MAP[w] = "Positive"
for w in negative_words: LABEL_MAP[w] = "Negative"
for w in neutral_words: LABEL_MAP[w] = "Neutral"

MODEL_PATH = os.path.join(ROOT, "model", "my_transformer")

# ── LOAD TRANSFORMER MODEL ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải mô hình Deep Learning...")
def load_custom_ai():
    if not os.path.exists(MODEL_PATH):
        return None, f"Không tìm thấy thư mục: {MODEL_PATH}"
    try:
        pipe = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)
        return pipe, None
    except Exception as e:
        return None, str(e)

my_ai, _model_err = load_custom_ai()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+|@\w+|#\w+", "", t)
    t = re.sub(r"[^a-zA-Z\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def map_dl_label(raw_label: str) -> str:
    return LABEL_MAP_INV.get(raw_label, "Neutral")

def predict(text: str) -> tuple[str, str | None]:
    if my_ai is None:
        return "⚠️ Model chưa tải", _model_err or "Chạy `python src/train_model.py` trước."
    text = text.strip()
    if not text:
        return "Neutral", "Văn bản rỗng."
    try:
        result = my_ai(text, truncation=True, max_length=512)[0]
        return map_dl_label(result["label"]), None
    except Exception as e:
        return "Error", str(e)

def predict_batch(texts: list[str]) -> tuple[list[str], str | None]:
    if my_ai is None:
        return [], _model_err or "Model chưa sẵn sàng."
    try:
        cleaned = [str(t) if str(t).strip() else " " for t in texts]
        # Thêm batch_size=16 giúp chống sập RAM khi phân tích hàng ngàn dòng
        preds = my_ai(cleaned, truncation=True, max_length=512, batch_size=16)
        return [map_dl_label(p["label"]) for p in preds], None
    except Exception as e:
        return [], str(e)

def show_sentiment_result(res: str, container=None):
    target = container or st
    col = CLR.get(res, "#6B7280")
    target.markdown(
        f'<div style="background:{col};color:#fff;padding:1.2rem;'
        f'border-radius:10px;text-align:center;font-size:1.4rem;'
        f'font-weight:700;margin-top:.5rem;">Kết quả: {res}</div>',
        unsafe_allow_html=True,
    )

def kpi(col, val, label):
    col.markdown(f'<div class="kpi-box"><div class="kpi-val">{val}</div><div class="kpi-lbl">{label}</div></div>', unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)

def wordcloud_img(text: str, bg="#1B2A4A") -> str:
    wc = WordCloud(width=700, height=320, background_color=bg, colormap="cool", max_words=80).generate(text or "no data")
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes: bytes, fname: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
    df.drop(columns=["Target"], errors="ignore", inplace=True)

    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        st.error(f"❌ File thiếu cột: {miss}")
        st.stop()

    df["Sentiment"] = df["Sentiment"].astype(str).str.strip()
    if df["Sentiment"].nunique() > 3:
        df["Sentiment"] = df["Sentiment"].map(LABEL_MAP).fillna("Neutral")
    df = df[df["Sentiment"].isin(["Positive", "Negative", "Neutral"])]

    # Ép kiểu an toàn (Fix bug ValueError khi file lỗi chứa chuỗi Text vào cột số)
    df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0)
    df["Retweets"] = pd.to_numeric(df["Retweets"], errors="coerce").fillna(0)

    if "Hashtag_Count" not in df.columns:
        df["Hashtag_Count"] = df["Hashtags"].apply(lambda x: len(re.findall(r"#\w+", str(x))))

    return compute_trending_score(df)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Sentiment Analyzer")
    st.markdown("---")
    uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])
    st.markdown("---")
    if my_ai is None:
        st.warning("⚠️ Model DL chưa sẵn sàng")
        if _model_err:
            with st.expander("Chi tiết lỗi"):
                st.code(_model_err)
    else:
        st.success("✅ Model Deep Learning đã tải")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📊 Social Media Sentiment Analysis Dashboard")
st.caption("Hệ thống phân tích cảm xúc & xu hướng mạng xã hội · Python + Deep Learning")

# ── HOME (chưa upload) ────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload file CSV ở sidebar để bắt đầu phân tích")
    st.markdown("---")
    st.markdown("### 🤖 Thử predict ngay (Deep Learning)")
    c1, c2 = st.columns([4, 1])
    txt = c1.text_area("Nhập câu:", height=80, placeholder="E.g. I love this amazing day!")
    c2.markdown("<br>", unsafe_allow_html=True)
    if c2.button("Phân tích", use_container_width=True):
        if txt.strip():
            res, err = predict(txt)
            if err and res == "Error":
                st.error(f"Lỗi phân tích: {err}")
            else:
                show_sentiment_result(res)
                if err: st.caption(f"⚠️ {err}")
        else:
            st.warning("Vui lòng nhập text.")
    if my_ai is None:
        st.warning("⚠️ Model chưa có. Chạy `python src/train_model.py` trước.")
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
raw = uploaded.read()
with st.spinner("Đang tải dữ liệu..."):
    try:
        df_full = load_and_prepare(raw, uploaded.name)
    except Exception as e:
        st.error(f"❌ Lỗi đọc file CSV: {e}")
        with st.expander("Debug traceback"):
            st.code(traceback.format_exc())
        st.stop()

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Bộ lọc")
    all_pf = sorted(df_full["Platform"].unique())
    sel_pf = st.multiselect("Platform", all_pf, default=all_pf)
    sel_st = st.multiselect("Sentiment", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])
    st.caption(f"Tổng: {len(df_full):,} dòng")

df = df_full[df_full["Platform"].isin(sel_pf if sel_pf else all_pf) & df_full["Sentiment"].isin(sel_st if sel_st else ["Positive", "Negative", "Neutral"])]

if df.empty:
    st.warning("Không có dữ liệu phù hợp bộ lọc.")
    st.stop()

# ── KPI STRIP ────────────────────────────────────────────────────────────────
k = st.columns(5)
n = len(df)
pos = (df["Sentiment"] == "Positive").sum()
neg = (df["Sentiment"] == "Negative").sum()
avg_likes = df['Likes'].mean()
avg_rts = df['Retweets'].mean()

kpi(k[0], f"{n:,}", "Tổng Posts")
kpi(k[1], f"{pos / n * 100:.1f}%", "😊 Positive")
kpi(k[2], f"{neg / n * 100:.1f}%", "😠 Negative")
kpi(k[3], f"{avg_likes:.0f}" if pd.notna(avg_likes) else "0", "Avg Likes")
kpi(k[4], f"{avg_rts:.0f}" if pd.notna(avg_rts) else "0", "Avg Retweets")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs(["📈 Tổng quan", "☁️ Word Cloud", "🔥 Trending", "⏰ Xu hướng", "🤖 Predict AI"])

# ════════════════════════ TAB 1: TỔNG QUAN ════════════════════════════════════
with t1:
    c1, c2 = st.columns(2)
    with c1:
        sec("Phân bố cảm xúc")
        cnt = df["Sentiment"].value_counts().reset_index()
        cnt.columns = ["Sentiment", "Count"]
        fig = px.pie(cnt, values="Count", names="Sentiment", color="Sentiment", color_discrete_map=CLR, hole=0.38)
        fig.update_layout(margin=dict(t=10, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Sentiment theo Platform")
        g = df.groupby(["Platform", "Sentiment"]).size().reset_index(name="Count")
        fig = px.bar(g, x="Platform", y="Count", color="Sentiment", barmode="group", color_discrete_map=CLR)
        fig.update_layout(margin=dict(t=10, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sec("Phân phối Likes")
        fig = px.histogram(df, x="Likes", color="Sentiment", nbins=40, barmode="overlay", opacity=0.7, color_discrete_map=CLR)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sec("Boxplot Retweets theo Sentiment")
        fig = px.box(df, x="Sentiment", y="Retweets", color="Sentiment", color_discrete_map=CLR, points=False)
        fig.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    sec("Heatmap — Giờ đăng bài × Sentiment")
    heat = df.groupby(["Hour", "Sentiment"]).size().reset_index(name="Count")
    pivot = heat.pivot(index="Hour", columns="Sentiment", values="Count").fillna(0)
    fig = px.imshow(pivot.T, aspect="auto", color_continuous_scale="teal", labels=dict(x="Giờ", y="Sentiment", color="Số bài"), text_auto=True)
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════ TAB 2: WORD CLOUD ═══════════════════════════════════
with t2:
    sec("Word Cloud — Nội dung theo cảm xúc")
    wc_col = st.columns(3)
    for i, lbl in enumerate(["Positive", "Negative", "Neutral"]):
        texts = df[df["Sentiment"] == lbl]["Text"].dropna()
        if len(texts) > 2000:
            texts = texts.sample(2000, random_state=42)
        corpus = " ".join(texts.astype(str).tolist())
        corpus = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", " ", corpus)
        with wc_col[i]:
            st.markdown(f"**{lbl}**")
            if corpus.strip():
                img_data = wordcloud_img(corpus)
                st.markdown(f'<img src="{img_data}" style="width:100%;border-radius:8px">', unsafe_allow_html=True)
            else:
                st.info("Không có dữ liệu")

# ════════════════════════ TAB 3: TRENDING ═════════════════════════════════════
with t3:
    c1, c2 = st.columns(2)
    with c1:
        sec("Top 15 Hashtag Trending")
        tags = top_hashtags(df, n=15)
        if not tags.empty:
            fig = px.bar(tags, x="TotalScore", y="Hashtag", orientation="h", color="Sentiment", color_discrete_map=CLR, labels={"TotalScore": "Trending Score"})
            fig.update_layout(yaxis_categoryorder="total ascending", margin=dict(t=10, b=10), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu hashtag")

    with c2:
        sec("Top 10 Posts Trending nhất")
        cols_show = ["Text", "Sentiment", "Platform", "Likes", "Retweets", "TrendingScore"]
        top10 = df.nlargest(10, "TrendingScore")[cols_show].copy()
        top10["TrendingScore"] = top10["TrendingScore"].round(5)
        top10["Text"] = top10["Text"].str[:60] + "..."
        st.dataframe(top10.reset_index(drop=True), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sec("Gợi ý — Giờ đăng bài hiệu quả nhất")
        bh = best_hour(df)
        if not bh.empty: # FIX Bẫy lỗi rỗng
            fig = px.bar(bh, x="Hour", y="AvgEngagement", color="AvgEngagement", color_continuous_scale="teal", labels={"Hour": "Giờ", "AvgEngagement": "Avg Engagement"})
            fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"🕐 Giờ đăng hiệu quả nhất: **{int(bh.iloc[0]['Hour'])}:00**")
        else:
            st.info("Chưa đủ dữ liệu giờ")

    with c4:
        sec("Gợi ý — Platform phù hợp nhất")
        bp = best_platform(df)
        if not bp.empty: # FIX Bẫy lỗi rỗng
            fig = px.bar(bp, x="Platform", y="AvgEngagement", color="Platform", labels={"AvgEngagement": "Avg Engagement"})
            fig.update_layout(margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"📱 Platform tốt nhất: **{bp.iloc[0]['Platform']}**")
        else:
            st.info("Chưa đủ dữ liệu Platform")

# ════════════════════════ TAB 4: XU HƯỚNG ════════════════════════════════════
with t4:
    sec("Xu hướng cảm xúc theo thời gian")
    sot = sentiment_over_time(df)
    if not sot.empty:
        fig = px.line(sot, x="YearMonth", y="Count", color="Sentiment", color_discrete_map=CLR, markers=True, labels={"YearMonth": "Tháng", "Count": "Số bài"})
        fig.update_layout(margin=dict(t=10, b=40), xaxis_tickangle=-45, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("Hoạt động theo giờ trong ngày")
        hc = df.groupby("Hour").size().reset_index(name="Count")
        fig = px.bar(hc, x="Hour", y="Count", color="Count", color_continuous_scale="teal", labels={"Hour": "Giờ", "Count": "Số bài"})
        fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Hoạt động theo tháng")
        mc = df.groupby("Month").size().reset_index(name="Count")
        fig = px.bar(mc, x="Month", y="Count", color="Count", color_continuous_scale="teal", labels={"Month": "Tháng", "Count": "Số bài"})
        fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════ TAB 5: PREDICT AI ═══════════════════════════════════
with t5:
    st.markdown("### 🤖 Predict cảm xúc (Deep Learning)")

    if my_ai is None:
        st.error("⚠️ Model Deep Learning chưa sẵn sàng. Chạy `python src/train_model.py` và đảm bảo có thư mục `model/my_transformer`.")
        if _model_err:
            with st.expander("Debug — chi tiết lỗi load model"): st.code(_model_err)
    else:
        st.markdown("#### Nhập text bất kỳ")
        c1, c2 = st.columns([4, 1])
        user_txt = c1.text_area("", height=90, key="pred_txt", placeholder="E.g. I hate football, but I hate ronaldo")
        c2.markdown("<br><br>", unsafe_allow_html=True)

        if c2.button("🔍 Phân tích", use_container_width=True):
            if user_txt.strip():
                res, err = predict(user_txt)
                if err and res == "Error":
                    st.error(f"Lỗi phân tích: {err}")
                    with st.expander("Debug traceback"): st.code(traceback.format_exc())
                else:
                    show_sentiment_result(res)
            else:
                st.warning("Vui lòng nhập text.")

        st.markdown("---")
        st.markdown("#### Predict hàng loạt từ CSV")
        batch = st.file_uploader("Upload CSV có cột 'Text':", type=["csv"], key="batch")
        if batch:
            try:
                bdf = pd.read_csv(batch)
            except Exception as e:
                st.error(f"❌ Không đọc được file CSV: {e}")
                st.stop()

            if "Text" not in bdf.columns:
                st.error("❌ File phải có cột 'Text'")
            else:
                if st.button("🚀 Chạy xử lý hàng loạt", use_container_width=True, key="batch_run"):
                    with st.spinner(f"AI đang xử lý {len(bdf):,} dòng... Việc này có thể mất chút thời gian."):
                        labels, err = predict_batch(bdf["Text"].astype(str).tolist())

                    if err:
                        st.error(f"Đã xảy ra lỗi khi dự đoán hàng loạt: {err}")
                        with st.expander("Debug traceback"): st.code(traceback.format_exc())
                    else:
                        bdf["Predicted_Sentiment"] = labels
                        st.success(f"✅ Đã predict xong {len(bdf):,} dòng bằng mô hình Transformer!")
                        st.dataframe(bdf[["Text", "Predicted_Sentiment"]].head(30), use_container_width=True)

                        cnt = bdf["Predicted_Sentiment"].value_counts().reset_index()
                        cnt.columns = ["Sentiment", "Count"]
                        fig = px.pie(cnt, values="Count", names="Sentiment", color="Sentiment", color_discrete_map=CLR, hole=0.35)
                        fig.update_layout(margin=dict(t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                        csv_out = bdf.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Tải kết quả CSV", csv_out, "deep_learning_results.csv", "text/csv", use_container_width=True)