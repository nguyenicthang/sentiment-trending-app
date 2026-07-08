import os
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer




def clean_text_with_nltk(text):
    if pd.isna(text):
        return ""
    

    text_str = str(text).lower()
    

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text_str)
    
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
    
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)



def extract_hashtags_to_list(text):
    if pd.isna(text):
        return []

    return re.findall(r'#\w+', str(text))


def main():
    input_file = os.path.join("data", "processed", "cleaned_data.csv")
    output_file = os.path.join("data", "processed", "featured_data.csv")
    
    if not os.path.exists(input_file):
        print(f" Không tìm thấy file '{input_file}'!")
        return
        
    print(f" Đang đọc dữ liệu từ: {input_file}")
    df = pd.read_csv(input_file)
    

    if 'Text' not in df.columns or 'Hashtags' not in df.columns:
        print(" File dữ liệu thiếu cột 'Text' hoặc 'Hashtags'!")
        print(f"Các cột hiện tại: {df.columns.tolist()}")
        return
        

    print(" Bước 1: Dùng Regex bóc tách cột Hashtags thành dạng list...")
    df['Hashtag_List'] = df['Hashtags'].apply(extract_hashtags_to_list)
    
    print(" Bước 2: Đếm số lượng hashtag từ list để tạo cột Hashtag_Count...")
    df['Hashtag_Count'] = df['Hashtag_List'].apply(len)
    

    print(" Bước 3: Chuẩn hóa Text (lowercase, xóa dấu câu, xóa Stopwords)...")
    df['Clean_Text'] = df['Text'].apply(clean_text_with_nltk)

    print(f" Đang xuất kết quả ra: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(" HOÀN THÀNH XỬ LÝ ĐÚNG THEO YÊU CẦU CỦA LEADER!")

if __name__ == "__main__":
    main()