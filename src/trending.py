"""
trending.py — Thắng (Tech Lead)
Module tính Trending Score + các hàm phân tích cho app.py
Không chạy trực tiếp — import vào app.py
"""
import re
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
W1, W2, W3 = 0.4, 0.4, 0.2   # trọng số Likes, Retweets, Hashtag_Count
LAMBDA      = 1.5              # time decay exponent

# ── HELPER ────────────────────────────────────────────────────────────────────
def _norm(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-9)

def _hours_elapsed(df: pd.DataFrame) -> pd.Series:
    """Tính giờ đã trôi qua từ lúc đăng đến hiện tại."""
    if "Hours_Elapsed" in df.columns:
        return df["Hours_Elapsed"].fillna(720).clip(lower=0)
    if "Timestamp" in df.columns:
        dt = pd.to_datetime(df["Timestamp"], format="mixed", errors="coerce")
        elapsed = (pd.Timestamp.now() - dt).dt.total_seconds() / 3600
        return elapsed.fillna(720).clip(lower=0)
    if {"Year","Month","Day","Hour"}.issubset(df.columns):
        dt = pd.to_datetime(df[["Year","Month","Day","Hour"]])
        elapsed = (pd.Timestamp.now() - dt).dt.total_seconds() / 3600
        return elapsed.fillna(720).clip(lower=0)
    return pd.Series(720.0, index=df.index)

# ── PUBLIC API ────────────────────────────────────────────────────────────────
def compute_trending_score(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột TrendingScore vào DataFrame."""
    df = df.copy()
    if "Hashtag_Count" not in df.columns:
        df["Hashtag_Count"] = df.get("Hashtags", pd.Series("")).apply(
            lambda x: len(re.findall(r"#\w+", str(x)))
        )
    df["_h_elapsed"] = _hours_elapsed(df)
    df["TrendingScore"] = (
        W1 * _norm(df["Likes"].fillna(0)) +
        W2 * _norm(df["Retweets"].fillna(0)) +
        W3 * _norm(df["Hashtag_Count"].fillna(0))
    ) / (df["_h_elapsed"] + 2) ** LAMBDA
    df.drop(columns=["_h_elapsed"], inplace=True)
    return df

def top_hashtags(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Top N hashtag theo tổng TrendingScore (Đã tối ưu Vectorization siêu tốc)."""
    if df.empty or "Hashtags" not in df.columns:
        return pd.DataFrame(columns=["Hashtag", "TotalScore", "Count", "Sentiment"])

    # Chỉ copy 3 cột cần thiết để tiết kiệm RAM
    tmp = df[["Hashtags", "Sentiment"]].copy()
    tmp["TrendingScore"] = df["TrendingScore"] if "TrendingScore" in df.columns else 1.0

    # Dùng regex tìm tất cả hashtag và cho vào 1 list
    tmp["Hashtag"] = tmp["Hashtags"].astype(str).str.findall(r"#\w+")
    
    # Tuyệt chiêu Explode: Tách 1 list thành nhiều dòng ngay lập tức (không cần vòng lặp for)
    tmp = tmp.explode("Hashtag")
    
    # Lọc bỏ các dòng không chứa hashtag
    tmp = tmp.dropna(subset=["Hashtag"])
    tmp["Hashtag"] = tmp["Hashtag"].astype(str).str.lower()

    if tmp.empty:
        return pd.DataFrame(columns=["Hashtag", "TotalScore", "Count", "Sentiment"])

    # Gom nhóm và tính toán cực nhanh bằng C++ core của Pandas
    return (
        tmp.groupby("Hashtag")
        .agg(
            TotalScore=("TrendingScore", "sum"),
            Count=("TrendingScore", "count"),
            # Lấy cảm xúc xuất hiện nhiều nhất (mode)
            Sentiment=("Sentiment", lambda x: x.mode()[0] if not x.mode().empty else "Neutral")
        )
        .reset_index()
        .sort_values("TotalScore", ascending=False)
        .head(n)
    )
def best_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Giờ đăng bài có engagement cao nhất."""
    return (
        df.groupby("Hour")[["Likes","Retweets"]]
        .mean()
        .assign(AvgEngagement=lambda x: x["Likes"] + x["Retweets"])
        .sort_values("AvgEngagement", ascending=False)
        .reset_index()
    )

def best_platform(df: pd.DataFrame) -> pd.DataFrame:
    """Platform có engagement cao nhất."""
    return (
        df.groupby("Platform")[["Likes","Retweets"]]
        .mean()
        .assign(AvgEngagement=lambda x: x["Likes"] + x["Retweets"])
        .sort_values("AvgEngagement", ascending=False)
        .reset_index()
    )

def sentiment_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Số post theo Year-Month và Sentiment."""
    grp = df.groupby(["Year","Month","Sentiment"]).size().reset_index(name="Count")
    grp["YearMonth"] = grp["Year"].astype(str) + "-" + grp["Month"].astype(str).str.zfill(2)
    return grp.sort_values("YearMonth")
