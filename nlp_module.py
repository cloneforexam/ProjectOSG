import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Sử dụng NLTK cho xử lý ngôn ngữ tự nhiên
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Tải dữ liệu từ vựng NLTK (tự động nếu chưa có)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class LinuxLogNLPModule:
    """
    MODULE NLP CHUYÊN DỤNG XỬ LÝ LOG SYSTEM LINUX (OSG203)
    Phụ trách: 2. Log Processing & NLP
    """

    def __init__(self, max_features=100, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english'
        )
        self.is_fitted = False

    def clean_and_normalize(self, text):
        """
        Bước 1 & 2: Text Cleaning & Normalization
        Biến đổi các giá trị động thành Token đại diện chuẩn hóa.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        # Biến đổi IP -> <IP>
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' <IP> ', text)
        # Biến đổi Hex/Address -> <HEX>
        text = re.sub(r'0x[0-9a-fA-F]+', ' <HEX> ', text)
        # Biến đổi PID/Port/Số -> <NUM>
        text = re.sub(r'\b\d+\b', ' <NUM> ', text)
        # Xóa dấu câu đặc biệt
        text = re.sub(r'[^a-z<>]', ' ', text)
        # Tách token và làm sạch khoảng trắng
        tokens = word_tokenize(text)
        cleaned_text = ' '.join(tokens)
        return cleaned_text

    def parse_syslog_line(self, line):
        """Phân tích cấu trúc bản ghi Syslog thô"""
        pattern = r'^([A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([\w\/\.-]+)(?:\[(\d+)\])?:\s+(.*)$'
        match = re.match(pattern, line)
        if match:
            timestamp, hostname, process, pid, message = match.groups()
            return {
                'timestamp': timestamp,
                'hostname': hostname,
                'process': process or 'system',
                'pid': pid or '0',
                'raw_message': message,
                'clean_message': self.clean_and_normalize(message)
            }
        return {
            'timestamp': 'N/A', 'hostname': 'localhost', 'process': 'unknown',
            'pid': '0', 'raw_message': line, 'clean_message': self.clean_and_normalize(line)
        }

    def process_raw_logs(self, raw_log_list):
        """Xử lý danh sách log thô thành DataFrame"""
        parsed_data = [self.parse_syslog_line(line) for line in raw_log_list]
        return pd.DataFrame(parsed_data)

    def fit_transform(self, df_logs):
        """Trích xuất đặc trưng TF-IDF và trả về ma trận đặc trưng"""
        corpus = df_logs['clean_message'].tolist()
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True
        
        feature_names = [f"tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        return tfidf_matrix, tfidf_df

    def export_processed_dataset(self, df_logs, tfidf_df, output_csv_path="processed_dataset.csv"):
        """Xuất BỘ DỮ LIỆU ĐÃ XỬ LÝ hoàn chỉnh"""
        final_df = pd.concat([df_logs[['timestamp', 'hostname', 'process', 'pid', 'raw_message', 'clean_message']], tfidf_df], axis=1)
        final_df.to_csv(output_csv_path, index=False)
        return final_df