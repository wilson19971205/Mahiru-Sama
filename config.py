from pathlib import Path

# Root dir
ROOT = Path(__file__).resolve().parent

# LLM settings
MODEL_PATH = str(ROOT / "module" / "llm_model" / "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
MMPROJ_PATH = str(ROOT / "module" / "llm_model" / "mmproj-Qwen3VL-8B-Instruct-F16.gguf")
IMAGE_PATH = str(ROOT / "image" / "test_image.jpg")
LLM_N_CTX = 2048
LLM_N_GPU_LAYERS = -1

# STT settings
STT_MODEL_PATH = str(ROOT / "module" / "stt_model" / "Qwen3-ASR-0.6B-bf16.gguf")
STT_MMPROJ_PATH = str(ROOT / "module" / "stt_model" / "mmproj-Qwen3-ASR-0.6B-bf16.gguf")
