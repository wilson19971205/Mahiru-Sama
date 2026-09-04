from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(ROOT / "model" / "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
MMPROJ_PATH = str(ROOT / "model" / "mmproj-Qwen3VL-8B-Instruct-F16.gguf")
IMAGE_PATH = str(ROOT / "image" / "test_image.jpg")