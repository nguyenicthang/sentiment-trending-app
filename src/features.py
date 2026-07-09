import os
import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Tải các gói NLTK cần thiết một cách tự động
for package in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
    except (LookupError, KeyError):
        nltk.download(package, quiet=True)

# DÙNG STOPWORDS TIẾNG ANH VÌ DATA LÀ TIẾNG ANH
ENGLISH_STOPWORDS = set(stopwords.words('english'))

def clean_text_with_nltk(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # 1. Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # 2. Xóa dấu câu
    tokens = word_tokenize(text) # 3. NLTK Tokenize
    # 4. Xóa Stopwords Tiếng Anh
    cleaned_tokens = [word for word in tokens if word not in ENGLISH_STOPWORDS] 
    return " ".join(cleaned_tokens)

def extract_hashtags_regex(text):
    if not isinstance(text, str):
        return ""
    # Tìm hashtag và ghép lại thành 1 chuỗi cách nhau bởi dấu phẩy (để lưu CSV không bị lỗi)
    hashtags = re.findall(r'#\w+', text)
    return ", ".join(hashtags) 

def main():
    input_file = os.path.join("data", "processed", "cleaned_data.csv")
    output_file = os.path.join("data", "processed", "featured_data.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file '{input_file}'! Bạn hãy kiểm tra lại xem đã pull code mới về chưa nhé.")
        return
        
    print(f"⏳ Đang đọc dữ liệu từ: {input_file}")
    df = pd.read_csv(input_file)
    
    if 'Text' not in df.columns:
        print("❌ Không tìm thấy cột 'Text' trong file cleaned_data.csv!")
        return
        
    print("✨ Dùng Regex bóc tách Hashtags & tạo cột Hashtag_Count...")
    # Tạo đúng cột 'Hashtags' để file analysis của TV4 đọc được
    df['Hashtags'] = df['Text'].apply(extract_hashtags_regex)
    
    # Đếm số lượng hashtag (nếu ô trống thì count = 0)
    df['Hashtag_Count'] = df['Hashtags'].apply(lambda x: len(x.split(',')) if x else 0)
    
    print("🧹 Dùng NLTK dọn sạch Text thành Clean_Text (Tiếng Anh)...")
    df['Clean_Text'] = df['Text'].apply(clean_text_with_nltk)
    
    print(f"💾 Đang xuất kết quả ra: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("✅ HOÀN THÀNH XỬ LÝ DỨT ĐIỂM!")

if __name__ == "__main__":
    main()