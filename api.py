"""
=============================================================================
 api.py - REST API (Flask) CHO GIAO DIỆN VÀ CÁC MODULE KHÁC (OPTIMIZED)
=============================================================================
"""
import logging
import os
import threading
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

import datahandle as dh

dh.setup_logging()
log = logging.getLogger("api")

app = Flask(__name__)
app.json.ensure_ascii = False    
app.json.sort_keys = False       

_state = {"es": None, "bundle": None}
_state_lock = threading.Lock()

def _get_es():
    with _state_lock:
        if _state["es"] is None:
            _state["es"] = dh.get_client()
        return _state["es"]

def _get_bundle():
    if _state["bundle"] is None:
        _state["bundle"] = dh.load_bundle()
    return _state["bundle"]

def _json_body(allow_empty=False):
    if allow_empty and not request.get_data(cache=True).strip():
        return {}
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise dh.InvalidInputError("Body phải là JSON dạng object, ví dụ: {\"logs\": [\"...\"]}.")
    return body

def _as_int(value, name, default=None, minimum=None):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise dh.InvalidInputError(f"'{name}' phải là số nguyên.")
    try:
        number = int(value)
    except ValueError as e:
        raise dh.InvalidInputError(f"'{name}' phải là số nguyên (nhận được {value!r}).") from e
    if minimum is not None and number < minimum:
        raise dh.InvalidInputError(f"'{name}' phải >= {minimum}.")
    return number

def _as_float(value, name, default):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise dh.InvalidInputError(f"'{name}' phải là số.")
    return float(value)

def _as_bool(value, name, default):
    if value is None:
        return default
    if not isinstance(value, bool):
        raise dh.InvalidInputError(f"'{name}' phải là true hoặc false.")
    return value

@app.errorhandler(dh.BackendError)
def handle_backend_error(exc):
    log.warning("%s: %s", type(exc).__name__, exc.message)
    return jsonify({"error": type(exc).__name__, "message": exc.message}), exc.status_code

@app.errorhandler(Exception)
def handle_unexpected(exc):
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.name, "message": exc.description}), exc.code
    log.exception("Lỗi không lường trước")
    return jsonify({"error": "InternalError", "message": "Lỗi nội bộ của backend, xem log để biết chi tiết."}), 500

@app.get("/")
def index():
    return jsonify({
        "service": "Linux Log Anomaly Detection - Backend API",
        "endpoints": ["GET /health", "POST /analyze", "GET /results", "POST /predict", "POST /upload"],
    })

@app.get("/health")
def health():
    bundle = _state["bundle"]
    if bundle is None:
        try:
            bundle = _get_bundle()
        except dh.ModelNotReadyError:
            bundle = None
    info = {
        "status": "ok", "api": "ok", "elasticsearch": "ok", "es_version": None,
        "model_loaded": bundle is not None,
        "model_trained_at": bundle["trained_at"] if bundle else None,
        "logs_index": dh.INDEX_LOGS, "results_index": dh.INDEX_RESULT,
    }
    try:
        es = _get_es()
        with dh.es_guard("kiểm tra sức khỏe"):
            info["es_version"] = es.info()["version"]["number"]
    except dh.BackendError as e:
        info.update({"status": "degraded", "elasticsearch": "unreachable", "message": e.message})
    return jsonify(info)

@app.post("/upload")
def upload():
    if 'file' not in request.files:
        raise dh.InvalidInputError("Không tìm thấy file trong request.")
        
    file = request.files['file']
    if file.filename == '':
        raise dh.InvalidInputError("Tên file trống.")
        
    try:
        content = file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        raise dh.InvalidInputError(f"Lỗi đọc file: {str(e)}")
        
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines:
        raise dh.InvalidInputError("File trống hoặc không có nội dung hợp lệ.")
        
    es = _get_es()
    
    def doc_generator():
        for line in lines:
            yield {
                "_index": dh.INDEX_LOGS,
                "_source": {"message": line}
            }
         
    try:
        with dh.es_guard("tải file log lên"):
            success_count = 0
            for success, info in streaming_bulk(es, doc_generator(), chunk_size=5000, raise_on_error=False):
                if success:
                    success_count += 1
            es.indices.refresh(index=dh.INDEX_LOGS)
    except Exception as e:
         log.exception("Lỗi khi ghi file upload vào ES")
         raise dh.BackendError(500, "UploadError", f"Lỗi ghi database: {str(e)}")
         
    return jsonify({
        "message": "Tải file thành công", 
        "filename": file.filename,
        "total_lines_uploaded": success_count
    })

@app.post("/analyze")
def analyze():
    body = _json_body(allow_empty=True)
    max_docs = _as_int(body.get("max_docs"), "max_docs", default=None, minimum=1)
    contamination = _as_float(body.get("contamination"), "contamination", dh.DEFAULT_CONTAMINATION)
    retrain = _as_bool(body.get("retrain"), "retrain", True)
    write_back = _as_bool(body.get("write_back"), "write_back", True)
 
    dh.validate_params(max_docs, contamination)
    es = _get_es() 
    result = dh.run_pipeline(
        es, max_docs=max_docs, contamination=contamination, retrain=retrain, write_back=write_back,
        bundle=None if retrain else _get_bundle(),
    )
    _state["bundle"] = result["bundle"]
    return jsonify(result["summary"])
 
@app.get("/results")
def results():
    # Cho phép nhận limit tối đa 100000
    limit = _as_int(request.args.get("limit"), "limit", default=10000, minimum=1)
    if limit > 100000:
        limit = 100000
    offset = _as_int(request.args.get("offset"), "offset", default=0, minimum=0)
    filters = dict(status=request.args.get("status"), severity=request.args.get("severity"),
                   threat_type=request.args.get("threat_type"), src_ip=request.args.get("src_ip"),
                   limit=limit, offset=offset)
    dh.build_results_query(**filters)
    total, items = dh.search_results(_get_es(), **filters)
    return jsonify({"total": total, "limit": limit, "offset": offset, "items": items})
 
@app.post("/predict")
def predict():
    body = _json_body()
    items = dh.predict_lines(_get_bundle(), body.get("logs"))
    return jsonify({"model_trained_at": _state["bundle"]["trained_at"], "count": len(items), "items": items})
 
if __name__ == "__main__":
    app.run(host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", "8000")), debug=False)
