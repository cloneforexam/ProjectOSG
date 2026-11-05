# File: run_nlp_experiment.py
import pandas as pd
from nlp_module import LinuxLogNLPModule

# --- BỘ DỮ LIỆU LOG ĐẦU VÀO THỬ NGHIỆM (RAW DATASET) ---
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

def run_experiment():
    print("==========================================================")
    print("  KẾT QUẢ THỬ NGHIỆM MODULE LOG PROCESSING & NLP (TF-IDF)")
    print("==========================================================\n")

    # 1. Khởi tạo Module NLP
    nlp_module = LinuxLogNLPModule(max_features=15, ngram_range=(1, 2))

    # 2. Xử lý Log & Parse
    df_parsed = nlp_module.process_raw_logs(dataset_raw_logs)

    # 3. Chạy TF-IDF Feature Extraction
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
    final_dataset = nlp_module.export_processed_dataset(df_parsed, tfidf_df, "processed_dataset.csv")
    print("\n" + "=" * 80)
    print(f"[THÀNH CÔNG] Đã xuất 'BỘ DỮ LIỆU ĐÃ XỬ LÝ' lưu tại: processed_dataset.csv")
    print(f"Kích thước bộ dữ liệu bàn giao: {final_dataset.shape[0]} dòng, {final_dataset.shape[1]} cột.")
    print("==========================================================")

if __name__ == "__main__":
    run_experiment()