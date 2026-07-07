import os
import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


for gia_tri in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{gia_tri}')
    except (LookupError, KeyError):
        nltk.download(gia_tri, quiet=True)


VIETNAMESE_STOPWORDS = set([
    'và', 'là', 'của', 'những', 'các', 'thì', 'mà', 'trong', 'để', 'có', 'cho', 'với', 'này'
])

def clean_text_with_nltk(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # 1. Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # 2. Xóa dấu câu
    tokens = word_tokenize(text) # 3. NLTK Tokenize
    cleaned_tokens = [word for word in tokens if word not in VIETNAMESE_STOPWORDS] # 4. Xóa Stopwords
    return " ".join(cleaned_tokens)

def extract_hashtags_regex(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'#\w+', text)

def main():
    # Đường dẫn file
    input_file = os.path.join("data", "processed", "cleaned_data.csv")
    output_file = os.path.join("data", "processed", "featured_data.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file '{input_file}'!")
        return
        
    print(f"⏳ Đang đọc dữ liệu từ: {input_file}")
    df = pd.read_csv(input_file)
    if 'Text' not in df.columns:
        print("❌ Không tìm thấy cột 'Text' trong file!")
        print(f"Các cột đang có là: {df.columns.tolist()}")
        return
    print("✨ Dùng Regex bóc tách Hashtags từ cột 'Text'...")
    df['Hashtag_List'] = df['Text'].apply(extract_hashtags_regex)
    df['Hashtag_Count'] = df['Hashtag_List'].apply(len)
    print("🧹 Dọn sạch dữ liệu văn bản bằng NLTK...")
    df['Clean_Text'] = df['Text'].apply(clean_text_with_nltk)
    df = df.drop(columns=['Hashtag_List'])
    print(f"💾 Đang xuất kết quả ra: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("✅ HOÀN THÀNH XỬ LÝ DỨT ĐIỂM!")

if __name__ == "__main__":
    main()