import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("data/raw/sentiment_dataset_full.csv")

# Hiển thị 5 dòng đầu tiên
#print(df.head(5))

# Kiểm tra Sentiment trước khi làm sạch nè :33
#print(df["Sentiment"].unique())

# Xóa khoảng trắng ở cột text
df["Text"] = df["Text"].astype(str).str.strip()

# Xóa khoảng trắng ở cột Sentiment
df["Sentiment"] = df["Sentiment"].astype(str).str.strip()

# Kiểm tra các nhãn
print("Số lượng nhãn:", df["Sentiment"].nunique())

# ==============
# CHUẨN HÓA PLATFORM
# ==============
df["Platform"] = (
    df["Platform"]
    .astype(str)
    .str.strip()
    .str.title()
)

print("\n === PLATFORM ===")
print(df["Platform"].unique())

# Kiểm tra dữ liệu còn thiếu
print("\n=== KIỂM TRA DỮ LIỆU CÒN THIẾU ===")
print(df.isnull().sum())

# Xóa NaN
df = df.dropna()

print("\n === SAU KHI XÓA NaN ===")
print(df.isnull().sum())

# Kiểm tra dữ liệu trùng lặp
print("\n === KIỂM TRA DỮ LIỆU TRÙNG LẶP ===")
print(df.duplicated().sum())

# ==============
# LABEL MAPPING
# ==============
positive = [
    "Positive","Happy","Happiness","Joy","Love","Amusement","Enjoyment",
    "Admiration","Affection","Awe","Surprise","Adoration",
    "Anticipation","Calmness","Excitement","Kind","Pride","Elation",
    "Euphoria","Contentment","Serenity","Gratitude","Hope",
    "Empowerment","Compassion","Tenderness","Enthusiasm","Fulfillment",
    "Reverence","Zest","Hopeful","Proud","Grateful","Empathetic",
    "Compassionate","Playful","Free-spirited","Inspired","Confident",
    "Thrill","Overjoyed","Inspiration","Motivation","Satisfaction",
    "Blessed","Appreciation","Confidence","Accomplishment","Wonderment",
    "Optimism","Enchantment","PlayfulJoy","Mindfulness","DreamChaser",
    "Harmony","Creativity","Radiance","Wonder","Rejuvenation","Coziness",
    "Adventure","Melodic","FestiveJoy","InnerJourney","Freedom","Dazzle",
    "Adrenaline","ArtisticBurst","CulinaryOdyssey","Resilience","Spark",
    "Marvel","Positivity","Kindness","Friendship","Success","Exploration",
    "Amazement","Romance","Captivation","Tranquility","Grandeur","Emotion",
    "Energy","Celebration","Charm","Ecstasy","Colorful","Hypnotic",
    "Connection","Iconic","Journey","Engagement","Touched","Triumph",
    "Heartwarming","Solace","Breakthrough","Joy in Baking",
    "Envisioning History","Imagination","Vibrancy","Mesmerizing",
    "Culinary Adventure","Winter Magic","Thrilling Journey",
    "Nature's Beauty","Celestial Wonder","Creative Inspiration",
    "Runway Creativity","Ocean's Freedom","Whispers of the Past",
    "Relief"
]
negative = [
    "Negative","Anger","Fear","Sadness","Disgust","Disappointed",
    "Bitter","Shame","Despair","Grief","Loneliness","Jealousy",
    "Resentment","Frustration","Boredom","Anxiety","Intimidation",
    "Helplessness","Envy","Regret","Melancholy","Bitterness",
    "Yearning","Fearful","Apprehensive","Overwhelmed","Jealous",
    "Devastated","Frustrated","Envious","Dismissive","Heartbreak",
    "Betrayal","Suffering","EmotionalStorm","Isolation",
    "Disappointment","LostLove","Exhaustion","Sorrow","Darkness",
    "Desperation","Ruins","Desolation","Loss","Heartache",
    "Obstacle","Pressure","Miscalculation","Challenge",
    "Embarrassed","Sad","Hate","Bad","Angry"
]
neutral = [
    "Neutral","Confusion","Curiosity","Indifference","Numbness",
    "Nostalgia","Ambivalence","Determination","Arousal",
    "Acceptance","Contemplation","Reflection","Intrigue",
    "Pensive","Immersion","Suspense","Excited"
]

# === BỔ SUNG CÁC NHÃN DÁN CÒN THIẾU SAU KHI ĐÃ KIỂM TRA TRƯỚC ĐÓ ! ===
positive.extend([
    "JoyfulReunion",
    "Elegance",
    "Whimsy",
    "Renewed Effort"
])

neutral.extend([
    "Bittersweet",
    "Sympathy",
    "Mischievous"
])

negative.extend([
    "Solitude"
])

# Tạo label_map
label_map = {}

for label in positive:
    label_map[label] = "Positive"

for label in negative:
    label_map[label] = "Negative"

for label in neutral:
    label_map[label] = "Neutral"

# Tạo cột Target
df["Target"] = df["Sentiment"].map(label_map)

# Thống kê TARGET
print("\n === THỐNG KÊ TARGET ===")
print(df["Target"].value_counts())

# Kiểm tra nhãn nào chưa có trong map
missing = df[df["Target"].isna()]["Sentiment"].unique()

print("\n === NHỮNG NHÃN CHƯA CÓ TRONG MAP ===")
print("Số nhãn chưa có trong map:", len(missing))
print("Các nhãn chưa có trong map:", missing)

# Lưu file đã làm sạch
df.to_csv(
    "data/cleaned_data.csv",
    index = False,
    encoding = "utf-8-sig"
)

print("Đã lưu file cleaned_data.csv thành công!")