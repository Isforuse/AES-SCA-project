import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime


class PrettyLogRunner:
    def __init__(self, target_script="cnn-model.py", log_dir="pretty_logs"):
        self.target_script = target_script
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.raw_log_path = self.log_dir / f"raw_{timestamp}.log"
        self.pretty_log_path = self.log_dir / f"pretty_{timestamp}.log"

        self.section_line = "=" * 80
        self.subsection_line = "-" * 60

    def classify_line(self, line: str):
        text = line.strip()

        if not text:
            return "BLANK", text

        if text.startswith("Traceback") or "ResourceExhaustedError" in text:
            return "ERROR", text

        if "OOM" in text or "Out of memory" in text or "ran out of memory" in text:
            return "ERROR", text

        if text.startswith("W0000") or "WARNING:" in text or "[WARN]" in text:
            return "WARN", text

        if text.startswith("I0000") or text.startswith("INFO:"):
            return "TF_INFO", text

        if "Epoch " in text and "/" in text:
            return "EPOCH", text

        if "metadata 對齊檢查通過" in text:
            return "DATA_OK", text

        if "未發現 null traces" in text:
            return "DATA_OK", text

        if "移除 null traces" in text:
            return "DATA_WARN", text

        if text.startswith("Byte ") and "class_weight:" in text:
            return "BYTE_WEIGHT", text

        if text.startswith("Byte ") and "true key:" in text:
            return "RESULT", text

        if text.startswith("Byte ") and "recovered key:" in text:
            return "RESULT", text

        if text.startswith("Byte ") and "final rank:" in text:
            return "RESULT", text

        if text.startswith("Byte ") and "accuracy:" in text:
            return "RESULT", text

        if text.startswith("Byte ") and "top-3 acc:" in text:
            return "RESULT", text

        if text.startswith("Byte ") and "macro F1:" in text:
            return "RESULT", text

        if text.startswith("Byte ") and "first rank-0 trace:" in text:
            return "RESULT", text

        if "開始處理 Target Byte" in text:
            return "SECTION", text

        if "完整 AES-128 Key Recovery 結果" in text:
            return "SECTION", text

        if "GPU 動態顯存配置已啟用" in text:
            return "INFO", text

        if "TensorFlow was not built with CUDA kernel binaries compatible" in text:
            return "GPU_NOTE", text

        if "Created device" in text and "GPU:0" in text:
            return "GPU_OK", text

        if "Loaded cuDNN version" in text:
            return "GPU_OK", text

        return "NORMAL", text

    def format_line(self, level: str, text: str):
        if level == "BLANK":
            return ""

        if level == "SECTION":
            return f"\n{self.section_line}\n{text}\n{self.section_line}"

        if level == "EPOCH":
            return f"\n[TRAIN] {text}"

        if level == "INFO":
            return f"[INFO] {text}"

        if level == "DATA_OK":
            return f"[DATA] {text}"

        if level == "DATA_WARN":
            return f"[DATA-WARN] {text}"

        if level == "WARN":
            return f"[WARN] {text}"

        if level == "ERROR":
            return f"[ERROR] {text}"

        if level == "GPU_NOTE":
            return f"[GPU-JIT] {text}"

        if level == "GPU_OK":
            return f"[GPU] {text}"

        if level == "BYTE_WEIGHT":
            return f"[CLASS-WEIGHT] {text}"

        if level == "RESULT":
            return f"[RESULT] {text}"

        if level == "TF_INFO":
            return f"[TF] {text}"

        return text

    def simplify_noise(self, text: str):
        text = re.sub(r"^I\d+\s+\d+:\d+\.\d+\s+\d+\s+", "", text)
        text = re.sub(r"^W\d+\s+\d+:\d+\.\d+\s+\d+\s+", "", text)
        return text

    def write_logs(self, raw_line: str, pretty_line: str):
        with self.raw_log_path.open("a", encoding="utf-8") as f:
            f.write(raw_line + "\n")

        with self.pretty_log_path.open("a", encoding="utf-8") as f:
            f.write(pretty_line + "\n")

    def print_header(self):
        header = [
            self.section_line,
            "Readable Training Log Wrapper",
            self.section_line,
            f"Target script : {self.target_script}",
            f"Raw log       : {self.raw_log_path}",
            f"Pretty log    : {self.pretty_log_path}",
            self.section_line,
            ""
        ]
        for line in header:
            print(line)

    def build_env(self):
        env = os.environ.copy()

        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
        env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

        return env

    def run(self):
        self.print_header()

        env = self.build_env()

        process = subprocess.Popen(
            [sys.executable, self.target_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        try:
            for line in process.stdout:
                raw_line = line.rstrip("\n")
                level, text = self.classify_line(raw_line)

                simplified = self.simplify_noise(text)
                pretty_line = self.format_line(level, simplified)

                print(pretty_line)
                self.write_logs(raw_line, pretty_line)

            process.wait()

        except KeyboardInterrupt:
            print("\n[WARN] 你手動中斷了執行。")
            process.terminate()
            process.wait()

        return process.returncode


def main():
    target_script = "cnn-model.py"
    if len(sys.argv) > 1:
        target_script = sys.argv[1]

    runner = PrettyLogRunner(target_script=target_script, log_dir="pretty_logs")
    exit_code = runner.run()

    print("\n" + "=" * 80)
    if exit_code == 0:
        print("[INFO] 程式執行完成")
    else:
        print(f"[ERROR] 程式結束，exit code = {exit_code}")
    print("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()