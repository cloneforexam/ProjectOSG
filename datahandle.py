import argparse
import contextlib
import ipaddress
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from itertools import islice

import joblib
import numpy as np
import pandas as pd
from elasticsearch import ApiError, Elasticsearch, NotFoundError, TransportError, helpers
from backend import (
    LinuxLogNLPModule,
    classify_threat,
    detect_threats,
    visualize_results,
)
 
log = logging.getLogger("datahandle")
 
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
INDEX_LOGS = os.getenv("ES_INDEX_LOGS", "linux-logs")
INDEX_RESULT = os.getenv("ES_INDEX_RESULT", "linux-logs-analyzed")
BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", "log_pipeline.joblib")
 
NLP__FEATURES = 500
DEFAULT_CONTAMINATION = 0.01
MIN_LOGS_TO_TRAIN = 10
MAX_RESULT_WINDOW = 100_000
MAX_PREDICT_LINES = 1000
MAX_LINE_CHARS = 5_000
NLP_MAX_FEATURES = 500  # Khai báo giá trị này để các hàm gọi không bị lỗi NameError
 
SEVERITY_LEVELS = ("Low", "Medium", "High")
SEVERITY_FALLBACK = (-0.10, -0.05)
MIN_SEVERITY_BY_THREAT = {"Brute-force SSH": "Medium", "Privilege Escalation": "Medium"}

RESULT_MAPPING = {
    "properties": {
        "source_doc_id": {"type": "keyword"},
        "log_event": {"type": "text"},
        "status": {"type": "keyword"},
        "threat_type": {"type": "keyword"},
        "severity": {"type": "keyword"},
        "anomaly_score": {"type": "float"},
        "src_ip": {"type": "ip"},
        "hostname": {"type": "keyword"},
        "process": {"type": "keyword"},
        "log_category": {"type": "keyword"},
        "syslog_timestamp": {"type": "keyword"},
        "detected_at": {"type": "date"},
        "last_analyzed_at": {"type": "date"},
    }
}
 
def setup_logging(level=logging.INFO):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

class BackendError(Exception):
    status_code = 500
    def __init__(self, message):
        super().__init__(message)
        self.message = message
 
class ESUnavailableError(BackendError):
    status_code = 503
 
class ESRequestError(BackendError):
    status_code = 502
 
class NoDataError(BackendError):
    status_code = 404
 
class DataFormatError(BackendError):
    status_code = 422
 
class InvalidInputError(BackendError):
    status_code = 422
 
class ModelNotReadyError(BackendError):
    status_code = 409
 
class PipelineBusyError(BackendError):
    status_code = 409
 
@contextlib.contextmanager
def es_guard(action):
    try:
        yield
    except BackendError:
        raise
    except NotFoundError as e:
        raise NoDataError(
            f"Không tìm thấy index khi {action}. Kiểm tra tên index và Logstash đã ghi dữ liệu chưa."
        ) from e
    except ApiError as e:
        raise ESRequestError(
            f"Elasticsearch từ chối yêu cầu khi {action} (HTTP {getattr(e, 'status_code', '?')}): "
            f"{getattr(e, 'message', e)}"
        ) from e
    except TransportError as e:
        raise ESUnavailableError(f"Mất kết nối Elasticsearch khi {action}: {e}") from e
 
def get_client(url=ES_URL):
    kwargs = {
        "request_timeout": 120,
        "max_retries": 3,
        "retry_on_timeout": True,
    }
    if ES_USER and ES_PASSWORD:
        kwargs["basic_auth"] = (ES_USER, ES_PASSWORD)
 
    es = Elasticsearch(url, **kwargs)
    try:
        reachable = es.ping()
    except Exception:
        reachable = False
    if not reachable:
        raise ESUnavailableError(f"Không kết nối được Elasticsearch tại {url}.")
    return es
 
def ensure_results_index(es, index=INDEX_RESULT):
    with es_guard("tạo index kết quả"):
        if es.indices.exists(index=index):
            return
        try:
            es.indices.create(index=index, mappings=RESULT_MAPPING)
            log.info("Đã tạo index kết quả '%s'.", index)
        except ApiError as e:
            if "resource_already_exists" not in str(e):
                raise
 
def fetch_logs(es, index=INDEX_LOGS, max_docs=None):
    rows = []
    with es_guard(f"đọc index '{index}'"):
        total = es.count(index=index)["count"]
        log.info("Index '%s' có %s document.", index, total)
        hits = helpers.scan(es, index=index, query={"query": {"match_all": {}}}, size=2000)
        try:
            for hit in islice(hits, max_docs):
                row = dict(hit["_source"])
                row["es_id"] = hit["_id"]
                rows.append(row)
        finally:
            close = getattr(hits, "close", None)
            if close:
                close()
    return pd.DataFrame(rows)
 
def to_training_dataframe(df_raw, drop_duplicates=True):
    if df_raw.empty:
        raise NoDataError("Index không có dữ liệu. Kiểm tra Logstash đã đọc xong chưa.")
    if "message" not in df_raw.columns:
        raise DataFormatError(f"Không thấy field 'message'. Các field hiện có: {list(df_raw.columns)}")
 
    df = df_raw.copy()
    if "log_message" in df.columns:
        primary = df["log_message"]
        missing = primary.isna() | (primary.astype(str).str.strip() == "")
        df["LogEvent"] = primary.where(~missing, df["message"])
    else:
        df["LogEvent"] = df["message"]
    df["LogEvent"] = df["LogEvent"].fillna("").astype(str)
    
    blank = df["LogEvent"].str.strip() == ""
    if blank.any():
        log.warning("Bỏ %d dòng log rỗng (thiếu nội dung).", int(blank.sum()))
        df = df[~blank]
 
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["message"])
        log.info("Loại %d dòng trùng, còn %d dòng.", before - len(df), len(df))
 
    if df.empty:
        raise NoDataError("Sau khi loại dòng rỗng/trùng không còn log nào để phân tích.")
    return df.reset_index(drop=True)
 
def train_model(df, contamination=DEFAULT_CONTAMINATION, max_features=500):
    if len(df) < MIN_LOGS_TO_TRAIN:
        raise NoDataError(f"Chỉ có {len(df)} log, cần ít nhất {MIN_LOGS_TO_TRAIN} log để huấn luyện mô hình.")
 
    train_df = df.sample(n=min(len(df), 30000), random_state=42) if len(df) > 30000 else df

    nlp = LinuxLogNLPModule(max_features=max_features, ngram_range=(1, 2))
    try:
        X_train = nlp.fit_features(train_df["LogEvent"].tolist())
    except ValueError as e:
        raise NoDataError(f"Không trích xuất được đặc trưng TF-IDF: {e}") from e
 
    model, _, anomalies = detect_threats(train_df, X_train, contamination_rate=contamination)
    thresholds = compute_severity_thresholds(anomalies["Anomaly_Score"])
 
    bundle = {
        "version": 1,
        "vectorizer": nlp.vectorizer,
        "model": model,
        "severity_thresholds": thresholds,
        "contamination": contamination,
        "nlp_max_features": max_features,
        "n_train": int(len(train_df)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    df_res, X = score_with_bundle(bundle, df)
    return bundle, df_res, X
 
def _nlp_from_bundle(bundle):
    nlp = LinuxLogNLPModule(max_features=bundle.get("nlp_max_features", NLP_MAX_FEATURES))
    nlp.vectorizer = bundle["vectorizer"]
    nlp.is_fitted = True
    return nlp
 
def score_with_bundle(bundle, df):
    nlp = _nlp_from_bundle(bundle)
    X = nlp.transform_features(df["LogEvent"].tolist())
    df = df.copy()
    df["AI_Prediction"] = bundle["model"].predict(X)
    df["Anomaly_Score"] = bundle["model"].decision_function(X)
    return df, X
 
def save_bundle(bundle, path=None):
    path = path or BUNDLE_PATH
    tmp = f"{path}.tmp"
    joblib.dump(bundle, tmp)
    os.replace(tmp, path)
 
def load_bundle(path=None):
    path = path or BUNDLE_PATH
    if not os.path.isfile(path):
        raise ModelNotReadyError(f"Chưa có mô hình ('{path}').")
    try:
        bundle = joblib.load(path)
    except Exception as e:
        raise ModelNotReadyError(f"File mô hình '{path}' bị hỏng: {e}.") from e
    return bundle

_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
 
def extract_ip(text):
    match = _IPV4.search(text or "")
    return match.group(0) if match else None
 
def compute_severity_thresholds(anomaly_scores):
    scores = np.asarray(anomaly_scores, dtype=float)
    if scores.size == 0:
        return SEVERITY_FALLBACK
    return float(np.quantile(scores, 1 / 3)), float(np.quantile(scores, 2 / 3))
 
def severity_from_score(score, thresholds, threat_type=None):
    high_t, medium_t = thresholds
    level = "High" if score <= high_t else "Medium" if score <= medium_t else "Low"
    floor = MIN_SEVERITY_BY_THREAT.get(threat_type)
    if floor and SEVERITY_LEVELS.index(level) < SEVERITY_LEVELS.index(floor):
        level = floor
    return level
 
def _clean(value):
    if value is None or isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return value
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value
 
def _text(value):
    value = _clean(value)
    return None if value is None else str(value)
 
def build_records(df, thresholds):
    records = []
    for row in df.to_dict("records"):
        is_anomaly = row["AI_Prediction"] == -1
        log_event = row["LogEvent"]
        score = float(row["Anomaly_Score"])
        threat = classify_threat(log_event) if is_anomaly else None
        records.append({
            "source_doc_id": _text(row.get("es_id")),
            "log_event": log_event,
            "status": "Anomaly" if is_anomaly else "Normal",
            "threat_type": threat,
            "severity": severity_from_score(score, thresholds, threat) if is_anomaly else None,
            "anomaly_score": round(score, 6),
            "src_ip": extract_ip(log_event),
            "hostname": _text(row.get("hostname")),
            "process": _text(row.get("process")),
            "log_category": _text(row.get("log_category")),
            "syslog_timestamp": _text(row.get("syslog_timestamp")),
        })
    return records
 
def summarize(records):
    anomalies = [r for r in records if r["status"] == "Anomaly"]
    by_severity = {level: 0 for level in reversed(SEVERITY_LEVELS)}
    by_threat = {}
    for r in anomalies:
        by_severity[r["severity"]] += 1
        by_threat[r["threat_type"]] = by_threat.get(r["threat_type"], 0) + 1
    return {
        "total_logs": len(records),
        "anomalies": len(anomalies),
        "normal": len(records) - len(anomalies),
        "by_severity": by_severity,
        "by_threat_type": by_threat,
    }
 
def write_results(es, records, index=INDEX_RESULT):
    ensure_results_index(es, index)
    now = datetime.now(timezone.utc).isoformat()
 
    def actions():
        for r in records:
            doc = {**r, "last_analyzed_at": now}
            yield {
                "_op_type": "update",
                "_index": index,
                "_id": r["source_doc_id"],
                "doc": doc,
                "upsert": {**doc, "detected_at": now},
            }
 
    with es_guard("ghi kết quả"):
        ok, errors = helpers.bulk(es, actions(), chunk_size=2000, raise_on_error=False)
        es.indices.refresh(index=index)
    failed = len(errors) if isinstance(errors, list) else int(errors or 0)
    log.info("Đã ghi %d bản ghi vào index '%s'.", ok, index)
    return ok, failed
 
def build_results_query(status=None, severity=None, threat_type=None, src_ip=None, limit=100, offset=0):
    if not 1 <= limit <= 100000:
        raise InvalidInputError("limit phải từ 1 đến 100000.")
    if offset < 0 or offset + limit > MAX_RESULT_WINDOW:
        raise InvalidInputError(f"offset phải >= 0 và offset + limit <= {MAX_RESULT_WINDOW}.")
 
    filters = []
    if status:
        if status.capitalize() not in ("Normal", "Anomaly"):
            raise InvalidInputError("status chỉ nhận 'Normal' hoặc 'Anomaly'.")
        filters.append({"term": {"status": status.capitalize()}})
    if severity:
        if severity.capitalize() not in SEVERITY_LEVELS:
            raise InvalidInputError("severity chỉ nhận 'Low', 'Medium' hoặc 'High'.")
        filters.append({"term": {"severity": severity.capitalize()}})
    if threat_type:
        filters.append({"term": {"threat_type": threat_type}})
    if src_ip:
        try:
            ipaddress.ip_address(src_ip)
        except ValueError as e:
            raise InvalidInputError(f"src_ip không phải địa chỉ IP hợp lệ: {src_ip!r}") from e
        filters.append({"term": {"src_ip": src_ip}})
 
    return {"bool": {"filter": filters}} if filters else {"match_all": {}}
 
def search_results(es, index=INDEX_RESULT, status=None, severity=None, threat_type=None,
                   src_ip=None, limit=100, offset=0):
    query = build_results_query(status, severity, threat_type, src_ip, limit, offset)
    try:
        with es_guard("đọc kết quả"):
            resp = es.search(
                index=index, query=query, size=limit, from_=offset, track_total_hits=True,
                sort=[{"detected_at": "desc"}, {"anomaly_score": "asc"}],
            )
    except NoDataError:
        return 0, []
    total = resp["hits"]["total"]["value"]
    return total, [hit["_source"] for hit in resp["hits"]["hits"]]

def predict_lines(bundle, lines):
    if not isinstance(lines, list) or not lines:
        raise InvalidInputError("'logs' phải là danh sách không rỗng các dòng log.")
    if len(lines) > MAX_PREDICT_LINES:
        raise InvalidInputError(f"Tối đa {MAX_PREDICT_LINES} dòng log mỗi lần gọi.")
    bad = [i for i, ln in enumerate(lines)
           if not isinstance(ln, str) or not ln.strip() or len(ln) > MAX_LINE_CHARS]
    if bad:
        raise InvalidInputError(f"Dòng log không hợp lệ tại vị trí: {bad[:10]}")
 
    nlp = _nlp_from_bundle(bundle)
    parsed = [nlp.parse_syslog_line(ln.strip()) for ln in lines]
    ok_format = [p["timestamp"] != "N/A" for p in parsed]
    df = pd.DataFrame({
        "LogEvent": [p["raw_message"] for p in parsed],
        "syslog_timestamp": [p["timestamp"] if ok else None for p, ok in zip(parsed, ok_format)],
        "hostname": [p["hostname"] if ok else None for p, ok in zip(parsed, ok_format)],
        "process": [p["process"] if ok else None for p, ok in zip(parsed, ok_format)],
    })
    df, _ = score_with_bundle(bundle, df)
    return build_records(df, bundle["severity_thresholds"])

_PIPELINE_LOCK = threading.Lock()
 
def validate_params(max_docs, contamination):
    if max_docs is not None and max_docs <= 0:
        raise InvalidInputError("max_docs phải là số nguyên dương (hoặc bỏ trống để lấy hết).")
    if not 0 < contamination <= 0.5:
        raise InvalidInputError("contamination phải nằm trong khoảng (0, 0.5].")
 
def run_pipeline(es, index=INDEX_LOGS, result_index=INDEX_RESULT, max_docs=None,
                 contamination=DEFAULT_CONTAMINATION, retrain=True, write_back=False,
                 plot=False, bundle=None):
    validate_params(max_docs, contamination)
    if not _PIPELINE_LOCK.acquire(blocking=False):
        raise PipelineBusyError("Đang có một lần phân tích khác chạy, hãy thử lại sau ít phút.")
    started = time.time()
    try:
        df_raw = fetch_logs(es, index, max_docs)
        df = to_training_dataframe(df_raw)
 
        if retrain:
            bundle, df_res, X = train_model(df, contamination)
            save_bundle(bundle)
            log.info("Đã huấn luyện trên %d log và lưu mô hình.", len(df))
        else:
            bundle = bundle or load_bundle()
            df_res, X = score_with_bundle(bundle, df)
            log.info("Đã chấm điểm %d log.", len(df))
 
        records = build_records(df_res, bundle["severity_thresholds"])
        written = failed = 0
        if write_back:
            written, failed = write_results(es, records, result_index)
 
        if plot:
            try:
                visualize_results(X, df_res)
            except Exception as e:
                log.warning("Không vẽ được biểu đồ: %s", e)
 
        summary = summarize(records)
        summary.update({
            "retrained": retrain,
            "model_trained_at": bundle["trained_at"],
            "contamination": bundle.get("contamination"),
            "written_to_es": written,
            "write_failed": failed,
            "duration_seconds": round(time.time() - started, 2),
        })
        return {"summary": summary, "records": records, "bundle": bundle}
    finally:
        _PIPELINE_LOCK.release()

def main(argv=None):
    setup_logging()
    parser = argparse.ArgumentParser(description="Lấy log từ Elasticsearch và chạy AI phát hiện bất thường",
                                     allow_abbrev=False)
    parser.add_argument("--url", default=ES_URL)
    parser.add_argument("--index", default=INDEX_LOGS)
    parser.add_argument("--result-index", default=INDEX_RESULT)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    parser.add_argument("--reuse-model", action="store_true")
    parser.add_argument("--write-back", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
 
    try:
        es = get_client(args.url)
        result = run_pipeline(
            es, index=args.index, result_index=args.result_index, max_docs=args.max_docs,
            contamination=args.contamination, retrain=not args.reuse_model,
            write_back=args.write_back, plot=args.plot,
        )
    except BackendError as e:
        log.error("%s", e.message)
        return 1
    except Exception:
        log.exception("Lỗi không lường trước")
        return 1
 
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0
 
if __name__ == "__main__":
    sys.exit(main())
