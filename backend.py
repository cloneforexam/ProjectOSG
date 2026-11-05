
#CÁCH CHẠY (mở terminal tại thư mục chứa file):
#  python backend.py              -> chạy tất cả (PHẦN 2 rồi PHẦN 3)
#  python backend.py nlp          -> chỉ chạy thử nghiệm NLP (file run_nlp_experiment.py cũ)
#  python backend.py train        -> chỉ chạy huấn luyện AI (file trainAI.py cũ)
#  python backend.py train --dataset duong_dan/file.csv   -> dùng file dataset khác
  
#THƯ VIỆN CẦN CÀI (nếu chưa có):
#  pip install pandas numpy scikit-learn matplotlib joblib nltk

import os            
import re
import sys
import json
import argparse

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# ---- NLTK: xử lý ngôn ngữ tự nhiên ----
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
# ---- scikit-learn: trích xuất đặc trưng & học máy ----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix
 
# ---- Hằng số cấu hình chung ----
# Gom tên file vào một chỗ để dễ sửa (file gốc ghi cứng tên trong từng hàm).
DATASET_PATH = "dataset"                         # file CSV log thật (cần có cột LogEvent và Activity)
PROCESSED_CSV_PATH = "processed_dataset.csv"     # file đầu ra của phần NLP
MODEL_PATH = "model/threat_detection_model.pkl"        # file mô hình đầu ra của phần train
PLOT_PATH = "isolation_forest_plot.png"          # file ảnh biểu đồ đầu ra của phần train
 
 
# ---- Tải dữ liệu NLTK (punkt / stopwords) ----
# Các bản NLTK mới (>= 3.8.2 / 3.9)
# đòi thêm 'punkt_tab', nếu thiếu thì word_tokenize() báo lỗi LookupError.
# Nên ở đây kiểm tra và tải cả 'punkt_tab'.
def _ensure_nltk_data():
    """Kiểm tra dữ liệu NLTK, thiếu thì tự tải (không báo lỗi nếu máy không có mạng)."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)            # thử tìm dữ liệu trên máy
        except LookupError:                          # không có -> thử tải về
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass                                 # không có mạng thì bỏ qua, đã có phương án dự phòng bên dưới

_NLTK_PREPARED = False
 
def _prepare_nltk():
    """[MỚI] Gọi _ensure_nltk_data() đúng 1 lần cho cả chương trình."""
    global _NLTK_PREPARED
    if not _NLTK_PREPARED:
        _ensure_nltk_data()
        _NLTK_PREPARED = True
 
_PUNKT_AVAILABLE = True   # chuyển thành False khi word_tokenize báo thiếu dữ liệu punkt
 
def _tokenize(text):
    """
    Tách token an toàn.
    Ưu tiên dùng word_tokenize của NLTK. Nếu máy thiếu dữ liệu punkt
    và không tải được (không có mạng) thì chuyển sang tách theo khoảng trắng.
    Cách dự phòng cho kết quả tương đương, vì đến bước này văn bản đã được làm sạch
    chỉ còn chữ thường, ký tự < > và khoảng trắng.
    """
    global _PUNKT_AVAILABLE
    if _PUNKT_AVAILABLE:
        try:
            return word_tokenize(text)
        except LookupError:
            _PUNKT_AVAILABLE = False     # từ lần sau đi thẳng sang cách tách theo khoảng trắng
    return text.split()
 
 
# =============================================================================
# PHẦN II: MODULE NLP XỬ LÝ LOG LINUX
# Nhiệm vụ: nhận log thô -> parse -> làm sạch/chuẩn hóa -> TF-IDF -> xuất dataset đã xử lý
# =============================================================================
class LinuxLogNLPModule:
    """
    MODULE NLP CHUYÊN DỤNG XỬ LÝ LOG SYSTEM LINUX (OSG203)
    """

    # II.1. Hàm khởi tạo: tạo bộ TF-IDF với các tham số cấu hình
    def __init__(self, max_features=100, ngram_range=(1, 2)):
        _prepare_nltk()                       # đảm bảo dữ liệu NLTK sẵn sàng (chỉ làm thật ở lần đầu)
        self.max_features = max_features      # số đặc trưng (từ/cụm từ) tối đa được giữ lại
        self.ngram_range = ngram_range        # (1, 2) = lấy cả từ đơn và cụm 2 từ liên tiếp
        self.vectorizer = TfidfVectorizer(
            lowercase=True,                   # chuyển chữ thường
            max_features=self.max_features,   # chỉ giữ N đặc trưng có tần suất cao nhất
            ngram_range=self.ngram_range,     # unigram + bigram
            stop_words='english'              # bỏ các từ vô nghĩa tiếng Anh (the, for, from...)
        )
        self.is_fitted = False                # cờ đánh dấu: đã học (fit) TF-IDF chưa
 
    # II.2. Làm sạch & chuẩn hóa một dòng log

    def clean_and_normalize(self, text):
        """
        Bước 1 & 2: Text Cleaning & Normalization
        Biến đổi các giá trị động thành Token đại diện chuẩn hóa.
        """
        if not isinstance(text, str):         # nếu không phải chuỗi (None, NaN...) -> trả chuỗi rỗng
            return ""
 
        text = text.lower()                   # đưa toàn bộ về chữ thường
 
     # dùng chữ THƯỜNG (<ip>, <hex>, <num>) để token được giữ lại.
 
        # Biến đổi IP -> <ip>   (ví dụ 192.168.1.200 -> <ip>)
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' <ip> ', text)
        # Biến đổi Hex/Address -> <hex>   (ví dụ 0x7ffd1234 -> <hex>)
        text = re.sub(r'0x[0-9a-fA-F]+', ' <hex> ', text)
        # Biến đổi PID/Port/Số -> <num>   (ví dụ 41100 -> <num>)
        text = re.sub(r'\b\d+\b', ' <num> ', text)
        # Xóa dấu câu đặc biệt: giữ lại chữ cái a-z và hai ký tự < > của token đại diện
        text = re.sub(r'[^a-z<>]', ' ', text)
 
        # Tách token và làm sạch khoảng trắng thừa
        tokens = _tokenize(text)              # dùng hàm _tokenize có phương án dự phòng
        cleaned_text = ' '.join(tokens)       # ghép lại thành 1 chuỗi, mỗi token cách nhau 1 dấu cách
 
        # word_tokenize tách "<ip>" thành 3 phần "<", "ip", ">" nên ghép lại cho đẹp: "< ip >" -> "<ip>"
        cleaned_text = re.sub(r'<\s*([a-z]+)\s*>', r'<\1>', cleaned_text)
        return cleaned_text
 
    # II.3. Phân tích cấu trúc một dòng Syslog thô

    def parse_syslog_line(self, line):
        """Phân tích cấu trúc bản ghi Syslog thô"""
        # Mẫu syslog:  <Tháng Ngày Giờ>  <hostname>  <tiến trình>[pid]:  <nội dung>
        pattern = r'^([A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([\w\/\.-]+)(?:\[(\d+)\])?:\s+(.*)$'
        match = re.match(pattern, line)
        if match:                             # dòng log đúng định dạng syslog
            timestamp, hostname, process, pid, message = match.groups()
            return {
                'timestamp': timestamp,
                'hostname': hostname,
                'process': process or 'system',   # thiếu tên tiến trình -> 'system'
                'pid': pid or '0',                # thiếu PID -> '0'
                'raw_message': message,           # nội dung gốc
                'clean_message': self.clean_and_normalize(message)   # nội dung đã chuẩn hóa
            }
        # Dòng log không đúng định dạng -> dùng giá trị mặc định, coi cả dòng là nội dung
        return {
            'timestamp': 'N/A', 'hostname': 'localhost', 'process': 'unknown',
            'pid': '0', 'raw_message': line, 'clean_message': self.clean_and_normalize(line)
        }
 
    # II.4. Xử lý cả danh sách log thô -> DataFrame

    def process_raw_logs(self, raw_log_list):
        """Xử lý danh sách log thô thành DataFrame"""
        parsed_data = [self.parse_syslog_line(line) for line in raw_log_list]  # parse từng dòng
        return pd.DataFrame(parsed_data)      # mỗi dòng log = 1 hàng của bảng
 
    # II.5. Trích xuất đặc trưng TF-IDF

    def fit_transform(self, df_logs):
        """Trích xuất đặc trưng TF-IDF và trả về ma trận đặc trưng"""
        corpus = df_logs['clean_message'].tolist()               # danh sách các câu đã làm sạch
        tfidf_matrix = self.vectorizer.fit_transform(corpus)     # học từ vựng + tính TF-IDF
        self.is_fitted = True
 
        # Đặt tên cột dạng tfidf_<từ> và chuyển ma trận thưa sang DataFrame thường
        feature_names = [f"tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        return tfidf_matrix, tfidf_df
 
    # II.5b. TF-IDF trực tiếp từ danh sách chuỗi (dùng cho backend/API)
    
    def fit_features(self, texts):
        """Làm sạch + chuẩn hóa từng dòng, HỌC từ vựng TF-IDF, trả về ma trận thưa (dùng lúc huấn luyện)."""
        cleaned = [self.clean_and_normalize(t) for t in texts]
        matrix = self.vectorizer.fit_transform(cleaned)
        self.is_fitted = True
        return matrix
 
    def transform_features(self, texts):
        """Biến log MỚI thành vector bằng từ vựng đã học (KHÔNG học lại) -> dùng để dự đoán."""
        if not self.is_fitted:
            raise RuntimeError("TF-IDF chưa được fit: hãy gọi fit_features() hoặc nạp vectorizer đã lưu.")
        cleaned = [self.clean_and_normalize(t) for t in texts]
        return self.vectorizer.transform(cleaned)
 
    # II.6. Xuất bộ dữ liệu đã xử lý ra CSV

    def export_processed_dataset(self, df_logs, tfidf_df, output_csv_path=PROCESSED_CSV_PATH):
        """Xuất BỘ DỮ LIỆU ĐÃ XỬ LÝ hoàn chỉnh"""
        # Ghép các cột thông tin log (timestamp, hostname...) với các cột TF-IDF theo chiều ngang
        final_df = pd.concat(
            [df_logs[['timestamp', 'hostname', 'process', 'pid', 'raw_message', 'clean_message']], tfidf_df],
            axis=1
        )
        final_df.to_csv(output_csv_path, index=False)   # ghi ra file CSV, không ghi cột chỉ số
        return final_df
 
# PHẦN III: CHẠY THỬ NGHIỆM MODULE NLP

# III.1. BỘ DỮ LIỆU LOG ĐẦU VÀO THỬ NGHIỆM (RAW DATASET)
dataset_raw_logs = [
    # Log bình thường (Normal Logs)
    "Oct 24 10:00:01 server1 systemd[1]: Started Session 10 of user student.",
    "Oct 24 10:00:05 server1 systemd[1]: Starting Cleanup of Temporary Directories...",
    "Oct 24 10:05:12 server1 sshd[2012]: Accepted password for student from 192.168.1.15 port 54321 ssh2",
    "Oct 24 10:06:01 server1 CRON[2050]: (root) CMD (cd / && run-parts --report /etc/cron.hourly)",
 
    # Log tấn công / Bất thường (Anomaly / Threat Logs)
    "Oct 24 10:10:01 server1 sshd[3010]: Failed password for root from 192.168.1.200 port 41100 ssh2",
    "Oct 24 10:10:03 server1 sshd[3012]: Failed password for root from 192.168.1.200 port 41102 ssh2",
    "Oct 24 10:10:05 server1 sshd[3015]: Failed password for invalid user admin from 192.168.1.200 port 41105 ssh2",
    "Oct 24 10:12:30 server1 sudo: attacker : 3 incorrect password attempts ; TTY=pts/1 ; USER=root",
    "Oct 24 10:15:00 server1 kernel: [ 4512.89] Out of memory: Kill process 8841 (bad_script) score 950",
    "Oct 24 10:16:22 server1 sshd[3100]: Invalid user anonymous from 45.33.32.156 port 58912"
]
 
# Nhãn thật cho 10 log ở trên (4 log đầu là Normal, 6 log sau là tấn công).
# Dùng ở PHẦN 3 để có dữ liệu chạy thử khi máy chưa có file "dataset" thật.
DEMO_ACTIVITY_LABELS = ["Normal"] * 4 + ["Attack"] * 6
 
# III.2. HÀM CHẠY THỬ NGHIỆM
def run_experiment():
    print("==========================================================")
    print("  KẾT QUẢ THỬ NGHIỆM MODULE LOG PROCESSING & NLP (TF-IDF)")
    print("==========================================================\n")
 
    # 1. Khởi tạo Module NLP (giữ tối đa 15 đặc trưng, dùng unigram + bigram)
    nlp_module = LinuxLogNLPModule(max_features=15, ngram_range=(1, 2))
 
    # 2. Xử lý Log & Parse: log thô -> DataFrame có cột raw_message / clean_message
    df_parsed = nlp_module.process_raw_logs(dataset_raw_logs)
 
    # 3. Chạy TF-IDF Feature Extraction: trả về ma trận thưa và DataFrame số
    tfidf_mat, tfidf_df = nlp_module.fit_transform(df_parsed)
 
    # 4. Hiển thị Kết quả Thử nghiệm trên Màn hình
    print("[1] BẢNG KẾT QUẢ TIỀN XỬ LÝ VĂN BẢN (TEXT CLEANING & NORMALIZATION):")
    print("-" * 80)
    for idx, row in df_parsed.iterrows():
        print(f"Log #{idx+1} [Raw]:   {row['raw_message']}")
        print(f"       [Clean]: {row['clean_message']}\n")
 
    print("\n[2] TOP 15 ĐẶC TRƯNG (FEATURES) XUẤT VIỆN TỪ TF-IDF:")
    print("-" * 80)
    print(list(tfidf_df.columns))
 
    print("\n[3] MA TRẬN VECTOR ĐẶC TRƯNG TF-IDF (Dạng số phục vụ AI):")
    print("-" * 80)
    print(tfidf_df.round(3).to_string())
 
    # 5. Xuất Bộ dữ liệu đã xử lý ra file CSV
    final_dataset = nlp_module.export_processed_dataset(df_parsed, tfidf_df, PROCESSED_CSV_PATH)
    print("\n" + "=" * 80)
    print(f"[THÀNH CÔNG] Đã xuất 'BỘ DỮ LIỆU ĐÃ XỬ LÝ' lưu tại: {PROCESSED_CSV_PATH}")
    print(f"Kích thước bộ dữ liệu bàn giao: {final_dataset.shape[0]} dòng, {final_dataset.shape[1]} cột.")
    print("==========================================================")
 
# PHẦN IV: HUẤN LUYỆN AI PHÁT HIỆN MỐI ĐE DỌA
# Quy trình: đọc dữ liệu -> TF-IDF -> Isolation Forest -> phân loại -> đánh giá -> vẽ biểu đồ

# BƯỚC 1 & 2: ĐỌC DỮ LIỆU VÀ TIỀN XỬ LÝ
# Tách thành 2 hàm để dùng lại được phần TF-IDF cho cả dữ liệu từ file lẫn dữ liệu chạy thử:
#   - preprocess_dataframe()       : phần TF-IDF (tách ra từ hàm gốc)
#   - load_and_preprocess_data()   : đọc CSV rồi gọi preprocess_dataframe() (hàm gốc, chạy y như cũ)
def preprocess_dataframe(df):
    """Biến cột LogEvent của DataFrame thành ma trận đặc trưng TF-IDF."""
    print("\n2. Preprocessing logs (Simulating Part 2 via TF-IDF)...")
 
    # Thay giá trị rỗng (NaN) bằng chuỗi rỗng, nếu không TfidfVectorizer sẽ báo lỗi
    df["LogEvent"] = df["LogEvent"].fillna("").astype(str)
 
    vectorizer = TfidfVectorizer()                           # TF-IDF mặc định (khác phần NLP ở PHẦN 1)
    X_features = vectorizer.fit_transform(df["LogEvent"])    # mỗi log -> 1 vector số
    print(f"-> Done! Feature matrix shape: {X_features.shape}")
    return df, X_features
 
def load_and_preprocess_data(file_path):
    print("1. Loading data from file...")
    # Đọc file dataset, bỏ qua dòng lỗi
    df = pd.read_csv(file_path, on_bad_lines='skip')
 
    # Báo lỗi rõ ràng nếu file thiếu cột LogEvent (thay vì lỗi KeyError khó hiểu)
    if "LogEvent" not in df.columns:
        raise ValueError(
            f"File '{file_path}' không có cột 'LogEvent'. Các cột hiện có: {list(df.columns)}"
        )
    return preprocess_dataframe(df)
 
# Tạo dữ liệu chạy thử từ 10 log ở PHẦN 2, dùng khi chưa có file "dataset".
# Đây là chỗ nối PHẦN 1 (parse log bằng LinuxLogNLPModule) với PHẦN 3 (huấn luyện AI).
def build_demo_dataset():
    nlp_module = LinuxLogNLPModule()
    df_parsed = nlp_module.process_raw_logs(dataset_raw_logs)   # parse log thô
    return pd.DataFrame({
        "LogEvent": df_parsed["raw_message"],     # nội dung log (giống cột LogEvent của dataset thật)
        "Activity": DEMO_ACTIVITY_LABELS          # nhãn thật để đánh giá (giống cột Activity)
    })
 
# BƯỚC 3 & 4: HUẤN LUYỆN VÀ PHÁT HIỆN

def detect_threats(df, X_features, contamination_rate=0.01):
    print("\n3. Training AI Model (Isolation Forest)...")
    # contamination = tỉ lệ log bất thường dự kiến trong dữ liệu (0.01 = 1%)
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
    model.fit(X_features)                                    # huấn luyện (không cần nhãn - học không giám sát)
 
    print("\n4. Analyzing and Detecting Threats...")
    df['AI_Prediction'] = model.predict(X_features)          # 1 = bình thường, -1 = bất thường
    df['Anomaly_Score'] = model.decision_function(X_features)  # điểm càng THẤP càng bất thường
 
    anomalies = df[df['AI_Prediction'] == -1].copy()         # lọc riêng các log bị đánh dấu bất thường
    print(f"-> Complete! Detected {len(anomalies)} anomalous logs.")
 
    return model, df, anomalies
 
# BƯỚC BỔ SUNG: PHÂN LOẠI MỐI ĐE DỌA

def classify_threat(log_event):
    """
    Phân loại chi tiết các mối đe dọa và sự kiện hệ thống từ dòng log.
    """
    msg = str(log_event).lower()

    # 1. Nhận diện các sự kiện Filebeat / Log Collector (Tránh bị gom thành Unknown Anomaly)
    if "filebeat" in msg:
        if "info" in msg:
            return "Filebeat Info Heartbeat"
        return "Filebeat Service Activity"
    elif "journal" in msg or "rsyslogd" in msg:
        return "Log Collector & Daemon Event"

    # 2. Tấn công SSH & Brute-force
    elif any(kw in msg for kw in ["failed password", "invalid user", "authentication failure", "preauth", "too many authentication failures"]):
        return "SSH Brute-force / Login Attack"
        
    # 3. Leo thang đặc quyền (Sudo / Su)
    elif any(kw in msg for kw in ["sudo:", "su:", "not in the sudoers file", "incorrect password attempt", "bad su"]):
        return "Privilege Escalation Attempt"

    # 4. Lệnh khả nghi & Chạy mã độc
    elif any(kw in msg for kw in ["wget ", "curl ", "base64", "chmod 777", "chown root:root", "nc -e", "/dev/tcp/"]):
        return "Suspicious Command Execution"

    # 5. Lỗi hệ thống nghiêm trọng & Tràn bộ nhớ
    elif any(kw in msg for kw in ["segfault", "out of memory", "oom-killer", "kernel panic", "hardware error", "core dumped"]):
        return "System Crash / Memory Error"

    # 6. Quét cổng & Lỗi mạng
    elif any(kw in msg for kw in ["connection refused", "refused connection", "unrecognized service", "invalid packet", "drop", "blocked", "connection", "port"]):
        return "Network & Connection Activity"

    # 7. Tiến trình định kỳ Cron / Systemd / Session
    elif "cron" in msg or "anacron" in msg or "session opened" in msg:
        return "Scheduled Task / Session Event"
    elif "systemd" in msg:
        return "Systemd Daemon Activity"

    # 8. Lỗi hệ thống chung (Error / Fail / Fatal)
    elif any(kw in msg for kw in ["error", "fatal", "fail", "critical"]):
        return "System Error / Warning"

    # 9. Nếu hoàn toàn không khớp các nhóm trên, trả về bất thường thống kê hành vi
    return "Behavioral Statistical Anomaly"

    
 
def apply_threat_classification(anomalies):
    print("\n--- THREAT CLASSIFICATION ---")
    # Gắn nhãn loại mối đe dọa cho từng log bất thường
    anomalies['Threat_Type'] = anomalies['LogEvent'].apply(classify_threat)
 
    # In 3 log bất thường nhất (Anomaly_Score thấp nhất)
    top_classified = anomalies.sort_values(by="Anomaly_Score").head(3)
    for index, row in top_classified.iterrows():
        print(f"Type: {row['Threat_Type']:<25} | Score: {row['Anomaly_Score']:.4f}")
 
    return anomalies
 
# BƯỚC 5 & 6: LƯU MÔ HÌNH VÀ ĐÁNH GIÁ

def export_and_evaluate(model, df):
    print("\n5. Exporting model for Backend integration...")
    joblib.dump(model, MODEL_PATH)                           # lưu mô hình để Backend nạp lại dùng
    print(f"-> Successfully saved '{MODEL_PATH}'!")
 
    print("\n6. Evaluating Model Accuracy...")
 
    # Nếu dataset không có cột Activity (nhãn thật) thì không thể đánh giá -> bỏ qua
    if 'Activity' not in df.columns:
        print("-> Dataset không có cột 'Activity' nên bỏ qua bước đánh giá.")
        return
 
    # Nhãn thật: 'Normal' -> 0, còn lại -> 1 (bất thường)
    y_true = [0 if str(act).strip() == 'Normal' else 1 for act in df['Activity']]
    # Nhãn dự đoán: AI_Prediction = 1 (bình thường) -> 0, -1 (bất thường) -> 1
    y_pred = [0 if p == 1 else 1 for p in df['AI_Prediction']]
 
    # Thêm labels=[0, 1]: nếu dữ liệu chỉ có 1 lớp thì file gốc báo lỗi
    # "Number of classes does not match size of target_names"
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=['Normal', 'Anomaly'], zero_division=0))
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
 
# BƯỚC 7: VẼ BIỂU ĐỒ 2D

def visualize_results(X_features, df):
    print("\n7. Generating 2D Visualization (PCA)...")
    # Lưu ý: thực chất là TruncatedSVD (hợp với ma trận thưa của TF-IDF), tên biến "pca" là của file gốc
    pca = TruncatedSVD(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_features)                     # nén vector nhiều chiều xuống 2 chiều
 
    normal_idx = df['AI_Prediction'] == 1                    # mặt nạ chọn log bình thường
    anomaly_idx = df['AI_Prediction'] == -1                  # mặt nạ chọn log bất thường
 
    plt.figure(figsize=(10, 6))
    plt.scatter(X_2d[normal_idx, 0], X_2d[normal_idx, 1], c='blue', alpha=0.5, label='Normal Logs', s=20)
    plt.scatter(X_2d[anomaly_idx, 0], X_2d[anomaly_idx, 1], c='red', alpha=0.8, label='Anomalies (Threats)', s=50, edgecolors='black')
 
    plt.title("AI Threat Detection - Isolation Forest Visualization (PCA)", fontsize=14, fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
 
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"-> Visualization saved as '{PLOT_PATH}'!")
    plt.close()   # giải phóng hình; nếu không, mỗi lần API gọi sẽ tích thêm 1 hình trong bộ nhớ
 
 
# -----------------------------------------------------------------------------
# Gom toàn bộ quy trình huấn luyện của trainAI.py (phần "main" cũ) thành một hàm
# -----------------------------------------------------------------------------
def run_train_pipeline(dataset_path=DATASET_PATH):
    if os.path.isfile(dataset_path):
        # Trường hợp có dataset thật: chạy y như trainAI.py gốc (contamination mặc định 1%)
        df_raw, X_feat = load_and_preprocess_data(dataset_path)
        contamination = 0.01
    else:
        # Trường hợp chưa có dataset: dùng 10 log thử nghiệm ở PHẦN 2 để chương trình vẫn chạy được.
        # Vì có 6/10 log là tấn công nên đặt contamination cao hơn (0.4) cho hợp lý.
        print(f"[!] Không tìm thấy file '{dataset_path}' -> dùng dữ liệu chạy thử (10 log ở PHẦN 2).")
        print("    Muốn dùng dữ liệu thật: đặt file CSV (có cột LogEvent, Activity) cạnh file này "
              f"với tên '{DATASET_PATH}', hoặc chạy với --dataset <đường_dẫn>.\n")
        df_raw, X_feat = preprocess_dataframe(build_demo_dataset())
        contamination = 0.4
 
    # Chạy mô hình
    trained_model, df_results, anomaly_logs = detect_threats(df_raw, X_feat, contamination_rate=contamination)
 
    # Phân loại
    anomaly_logs = apply_threat_classification(anomaly_logs)
 
    # Đánh giá & Lưu
    export_and_evaluate(trained_model, df_results)
 
    # Vẽ biểu đồ
    visualize_results(X_feat, df_results)
 
# PHẦN V: CHƯƠNG TRÌNH CHÍNH (MAIN EXECUTION)

def main():
    # Ép in tiếng Việt bằng UTF-8 để không lỗi trên terminal Windows (nếu hỗ trợ)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
 
    parser = argparse.ArgumentParser(description="Phát hiện bất thường trong log Linux (NLP + Isolation Forest)")
    parser.add_argument("mode", nargs="?", default="all", choices=["all", "nlp", "train"],
                        help="all = chạy cả hai (mặc định) | nlp = chỉ thử nghiệm NLP | train = chỉ huấn luyện AI")
    parser.add_argument("--dataset", default=DATASET_PATH,
                        help="đường dẫn file CSV log thật (cần cột LogEvent, Activity)")
    # parse_known_args: bỏ qua tham số lạ (ví dụ khi chạy trong Jupyter/Spyder) thay vì báo lỗi
    args, _unknown = parser.parse_known_args()
 
    if args.mode in ("all", "nlp"):
        run_experiment()
 
    if args.mode == "all":
        print("\n\n")                                        # ngăn cách hai phần cho dễ đọc
 
    if args.mode in ("all", "train"):
        run_train_pipeline(args.dataset)                     # chạy phần của trainAI.py
 
 
if __name__ == "__main__":
    main()
 