"""
train_model.py — (Deep Learning Edition - Hàng Real)
Fine-tune mô hình Transformer (DistilBERT) bằng PyTorch trên dữ liệu thực tế.
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Tắt cảnh báo của Hugging Face
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("="*60)
print("🚀 KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN DEEP LEARNING (PYTORCH) 🚀")
print("="*60)

# 1. ── ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ───────────────────────────────────────────────
print("📂 Đang nạp dữ liệu từ cleaned_data.csv...")
df = pd.read_csv("data/processed/cleaned_data.csv")

# ĐỂ CHẠY KỊP TRONG ĐÊM NAY BẰNG LAPTOP: Chỉ lấy 3000 dòng ngẫu nhiên.
# Mở khóa dòng dưới (thành None) nếu em chạy trên máy có Card Đồ Họa (GPU)
SAMPLE_SIZE = 3000 
if SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42)
    print(f"⚠️ Chế độ CPU: Đang train trên mẫu {SAMPLE_SIZE} dòng để kịp thời gian.")

# Map nhãn chữ sang số (PyTorch yêu cầu định dạng số)
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
df["Label"] = df["Sentiment"].map(LABEL_MAP)

# Loại bỏ các dòng bị lỗi NaN sau khi map
df = df.dropna(subset=["Label", "Text"])
texts = df["Text"].astype(str).tolist()
labels = df["Label"].astype(int).tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"📊 Tập Train: {len(train_texts)} câu | Tập Test: {len(val_texts)} câu")

# 2. ── KHỞI TẠO BỘ NÃO & TỪ ĐIỂN ─────────────────────────────────────────────
print("🧠 Đang tải cấu trúc DistilBERT...")
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 3. ── CHUYỂN ĐỔI DATA SANG PYTORCH TENSORS ──────────────────────────────────
print("⚙️ Đang Tokenize dữ liệu (Chuyển chữ thành Ma trận Tensor)...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Hàm tính độ chính xác trong lúc học
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}

# 4. ── CẤU HÌNH HUẤN LUYỆN LƯỚI ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,              # Học 2 vòng lặp qua toàn bộ dữ liệu
    per_device_train_batch_size=8,   # CPU tải 8 câu cùng lúc
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    report_to="none"                 # Tắt report online
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# KÍCH HOẠT QUÁ TRÌNH HỌC TẬP (Cập nhật Weights thực sự)
print("🔥 Bắt đầu quá trình Huấn luyện (Fine-tuning)... Vui lòng chờ...")
trainer.train()

# 5. ── LƯU THÀNH QUẢ ĐỂ WEB SỬ DỤNG ──────────────────────────────────────────
print("✅ Huấn luyện hoàn tất! Đang lưu mô hình do chính bạn tạo ra...")
save_path = "model/my_transformer"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"🎉 XUẤT XẮC! Mô hình Deep Learning Real 100% đã được lưu tại: {save_path}")