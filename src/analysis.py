import pandas as pd
from datetime import datetime

df = pd.read_csv("data/processed/cleaned_data.csv")   #đọc dữ liệu
print(df.columns)

#tạo cột datetime
df["Datetime"] = pd.to_datetime(          
    df[["Year", "Month", "Day", "Hour"]]
)
print(df["Datetime"].head())

#tính thời gian từ lúc đăng bài đến hiện tại
now = datetime.now()  
df["Hours_Elapsed"] = (
    now - df["Datetime"]
).dt.total_seconds() / 3600
print(df["Hours_Elapsed"].head())

#tính tổng likes và retweets theo từng Hashtag, từng Platform
summary = (                
    df.groupby(["Hashtags", "Platform"])[["Likes", "Retweets"]]
      .sum()
      .reset_index()
)
print(summary.head())

df.to_csv("data/processed/analyzed_data.csv", index=False)
summary.to_csv("data/processed/groupby_summary.csv", index=False)
print("data analyst hoàn thành")

