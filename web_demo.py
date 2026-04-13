import os
import sys
import html
import traceback
import tempfile
import importlib.util
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Add src to path so relative imports work for model and config
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Dynamic import for numeric-start filename module
INFERENCE_PATH = SRC_DIR / "4_inference.py"
inference = None
inference_load_error = None
try:
    spec = importlib.util.spec_from_file_location("guitar_inference", INFERENCE_PATH)
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
except Exception as err:
    inference_load_error = err

from config import SAMPLE_RATE, FEATURE_TYPE
import paths


def import_model_class():
    try:
        from model import GuitarTranscriberCNN
    except Exception as err:
        raise RuntimeError(f"Không thể import model class từ model.py: {err}") from err
    return GuitarTranscriberCNN

HOST = "127.0.0.1"
PORT = 8501

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GUITAR-AI Web Demo</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
        .container {{ max-width: 900px; margin: auto; padding: 24px; background: white; border-radius: 12px; box-shadow: 0 12px 30px rgba(0,0,0,0.08); }}
        h1 {{ margin-bottom: 0.4em; }}
        .notice {{ margin: 14px 0; padding: 12px 14px; border-left: 4px solid #0078d4; background: #eef5ff; }}
        .warning {{ border-left-color: #d97706; background: #fff7ed; }}
        input[type=file] {{ margin: 12px 0; }}
        button {{ padding: 10px 16px; font-size: 1rem; border: none; border-radius: 6px; background: #0078d4; color: white; cursor: pointer; }}
        button:hover {{ background: #005a9e; }}
        pre {{ white-space: pre-wrap; word-break: break-word; background: #111; color: #dcdcdc; padding: 16px; border-radius: 8px; overflow-x: auto; }}
        .footer {{ margin-top: 24px; font-size: 0.95rem; color: #555; }}
    </style>
</head>
    <div class="container">
        <h1>GUITAR-AI Web Demo</h1>
        <p>Chọn file âm thanh (WAV hoặc MP3), sau đó nhấn nút để mô hình tạo tab guitar trên màn hình.</p>
        <div class="notice">
            <strong>Lưu ý:</strong> Máy chủ này sử dụng mô hình và công thức có sẵn trong dự án của bạn.
        </div>

        <form method="POST" enctype="multipart/form-data">
            <label for="audio_file">File âm thanh:</label><br />
            <input type="file" id="audio_file" name="audio_file" accept="audio/*" required /><br />
            <button type="submit">Tạo tab</button>
        </form>
        {result}
        <div class="footer">
            <p>Mô hình dự kiến được lưu tại: <code>{model_path}</code></p>
            <p>Đang dùng cấu hình: <strong>{feature_type}</strong> với sample rate <strong>{sample_rate}</strong>.</p>
        </div>
    </div>
</body>
</html>
"""


def load_model():
    if inference_load_error is not None:
        raise RuntimeError(
            "Không thể tải module inference do lỗi môi trường hoặc PyTorch:\n"
            f"{inference_load_error}\n"
            "Hãy kiểm tra rằng PyTorch được cài đặt đúng và DLL phụ thuộc đã sẵn sàng."
        )
    GuitarTranscriberCNN = import_model_class()
    try:
        import torch
    except Exception as err:
        raise RuntimeError(
            "Không thể import torch. Vui lòng kiểm tra cài đặt PyTorch và driver GPU/Visual C++ trên Windows:\n"
            f"{err}"
        ) from err

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = paths.MODEL_DIR / "guitar_model.pth"
    if not model_path.exists():
        candidates = sorted(paths.MODEL_DIR.glob("*.pth"))
        if candidates:
            model_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Không tìm thấy điểm lưu mô hình trong {paths.MODEL_DIR}."
                " Hãy chạy training trước hoặc đặt file `.pth` vào thư mục này."
            )

    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model, device, model_path


def render_result(output_text, error=None):
    if error:
        content = f"<div class=\"notice warning\"><strong>Lỗi:</strong> {html.escape(error)}</div><pre>{html.escape(output_text)}</pre>"
    else:
        content = f"<div class=\"notice\"><strong>Kết quả:</strong></div><pre>{html.escape(output_text)}</pre>"
    return content


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class DemoHandler(BaseHTTPRequestHandler):
    model = None
    device = None
    model_path = None

    @classmethod
    def initialize_model(cls):
        if cls.model is None:
            cls.model, cls.device, cls.model_path = load_model()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_TEMPLATE.format(result="", model_path=self.model_path, feature_type=FEATURE_TYPE, sample_rate=SAMPLE_RATE).encode("utf-8"))

    def do_POST(self):
        self.initialize_model()
        result_html = ""

        try:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Yêu cầu phải là multipart/form-data")

            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            })

            upload = form["audio_file"] if "audio_file" in form else None
            if upload is None or not upload.filename:
                raise ValueError("Không tìm thấy file âm thanh tải lên.")

            suffix = Path(upload.filename).suffix.lower()
            if suffix not in [".wav", ".mp3", ".flac", ".ogg"]:
                raise ValueError("Chỉ hỗ trợ file WAV, MP3, FLAC hoặc OGG.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.file.read())
                tmp_path = Path(tmp.name)

            features = inference.extract_features(str(tmp_path))
            if features is None:
                raise RuntimeError("Không thể trích xuất đặc trưng từ file âm thanh.")

            raw_preds = inference.predict(self.model, features, self.device)
            if raw_preds is None:
                raise RuntimeError("Mô hình không thể tạo dự đoán từ file âm thanh.")

            decoder = inference.GuitarViterbi(threshold=0.85)
            final_binary = np.zeros_like(raw_preds)
            for s in range(6):
                path = decoder.decode_string(raw_preds[:, s, :])
                for t, state in enumerate(path):
                    if state < 21:
                        final_binary[t, s, state] = 1.0

            tab_text = inference.matrix_to_tab(final_binary, f"TRANSCRIBED: {upload.filename}")
            result_html = render_result(tab_text)
        except Exception as ex:
            error_msg = str(ex)
            stack = traceback.format_exc()
            result_html = render_result(stack, error=error_msg)
        finally:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_TEMPLATE.format(result=result_html, model_path=self.model_path, feature_type=FEATURE_TYPE, sample_rate=SAMPLE_RATE).encode("utf-8"))


if __name__ == "__main__":
    try:
        import torch
        import numpy as np
        import cgi
    except (ImportError, OSError) as err:
        print("Thiếu thư viện hoặc PyTorch không thể khởi tạo:", err)
        print("Hãy kiểm tra:")
        print("  1) Bạn đang dùng đúng môi trường Python/conda không?")
        print("  2) Cài lại PyTorch phù hợp với CPU hoặc GPU của bạn.")
        print("  3) Đã cài Visual C++ Redistributable và driver GPU tương thích chưa?")
        print("Ví dụ: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    print(f"Khởi động GUITAR-AI Web Demo tại http://{HOST}:{PORT}")
    print("Mở trình duyệt và tải file âm thanh lên để tạo tab.")
    server = ThreadedHTTPServer((HOST, PORT), DemoHandler)
    DemoHandler.initialize_model()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDừng web demo.")
        server.server_close()
