import base64
import collections
import io
import os
import sys
import time
import numpy as np
import opencc
import pyaudio
import scipy.io.wavfile as wavfile
from llama_cpp import Llama
# 順應你的硬體與環境，改回使用原本能動的 2.5 視覺 Handler
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

# 強制 Windows 終端機採用 UTF-8 編碼，防止印出中文字顯示亂碼
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class STT_Core:

    def __init__(self, config) -> None:

        self.config = config

        # 讀取 config 中的 GGUF 與投影檔路徑
        self.model_path = getattr(
            config, "STT_MODEL_PATH", "./module/stt_model/Qwen3-ASR-0.6B-bf16.gguf"
        )
        self.mmproj_path = getattr(
            config,
            "STT_MMPROJ_PATH",
            "./module/stt_model/mmproj-Qwen3-ASR-0.6B-bf16.gguf",
        )
        self.n_ctx = getattr(config, "STT_N_CTX", 2048)
        self.n_gpu_layers = getattr(config, "STT_N_GPU_LAYERS", -1)
        self.rate = getattr(config, "STT_SAMPLE_RATE", 16000)

        # 讀取全局繁簡轉換設定
        lang_format = getattr(config, "LANGUAGE_FORMAT", "s2twp")
        self.cc = opencc.OpenCC(lang_format)

        print("🔄 [STT] 正在常駐載入 Qwen3-ASR GGUF 核心 (2.5 Handler 模式)...")

        # 核心修正：使用你指定能 work 的 Qwen25VLChatHandler
        self.chat_handler = Qwen25VLChatHandler(
            clip_model_path=self.mmproj_path,
            verbose=getattr(config, "DEBUG_MODE", False),
        )

        self.llm = Llama(
            model_path=self.model_path,
            chat_handler=self.chat_handler,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=getattr(config, "DEBUG_MODE", False),
        )

        # 音訊控制初始化
        self.p = pyaudio.PyAudio()
        self.channels = 1
        self.chunk = 1024

        # 滾動緩衝區（保留最近 4 秒的聲音）
        self.audio_buffer = collections.deque(maxlen=int(self.rate / self.chunk * 4))
        self.stream = None
        self.is_listening = False

        print("✅ [STT] GGUF 語音核心常駐 GPU 成功！")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """內部麥克風收音監聽"""
        if self.is_listening:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_buffer.append(audio_data)
        return (None, pyaudio.paContinue)

    def start_listening(self):
        """開啟即時監聽與語音辨識循環"""
        if self.is_listening:
            return

        self.audio_buffer.clear()
        self.is_listening = True

        # 開啟麥克風管道
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._audio_callback,
        )

        print("\n🎤 【Qwen3-ASR GGUF】已開啟即時錄音監聽... (按下 Ctrl+C 結束測試)")
        self.stream.start_stream()

        try:
            while self.stream.is_active() and self.is_listening:
                time.sleep(0.4)  # 每 0.4 秒送驗一次

                if len(self.audio_buffer) == 0:
                    continue

                # 1. 拼接緩衝區內最近的音訊
                full_audio = np.concatenate(list(self.audio_buffer))

                # 📊 音量安全檢查，環境如果完全太安靜直接跳過推理（省 VRAM）
                rms = np.sqrt(np.mean(full_audio.astype(np.float32) ** 2))
                if rms < 100.0:
                    continue

                try:
                    # 2. 將記憶體中的音訊陣列轉為標準 WAV 檔案格式的二進位資料
                    wav_io = io.BytesIO()
                    wavfile.write(wav_io, self.rate, full_audio)
                    wav_bytes = wav_io.getvalue()

                    # 3. 轉化為 Base64（注意：這裡格式宣告為通用 octet-stream 偽裝）
                    audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
                    data_url = f"data:application/octet-stream;base64,{audio_base64}"

                    # 4. 關鍵偽裝：維持 type 為 "image_url"，強迫 Qwen25VLChatHandler 幫我們處理二進位資料
                    # 同時在文字前手動加上語音特殊 Token，喚醒語音 GGUF 模型
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "<|audio_bos|><|AUDIO|><|audio_eos|>請將這段語音轉寫為文字。",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                },
                            ],
                        }
                    ]

                    # 5. 呼叫建立對話 completion
                    response = self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=64,
                        temperature=0.0,
                    )

                    # 6. 解析輸出並進行繁體轉換
                    raw_text = response["choices"][0]["message"]["content"].strip()
                    tw_text = self.cc.convert(raw_text)
                    print("\n\n乾淨的文字: ", tw_text, "\n\n")
                    
                    if tw_text:
                        # 用 \r 達成動態更新同行的效果
                        print(f"\r💬 語音辨識結果: {tw_text}    ", end="", flush=True)

                except Exception as e:
                    pass

        except KeyboardInterrupt:
            self.stop_listening()

    def stop_listening(self):
        """停止監聽並關閉串流"""
        if not self.is_listening:
            return
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("\n🛑 [STT] 語音監聽已關閉。")

    def close(self):
        """全面釋放底層硬體資源"""
        self.stop_listening()
        self.p.terminate()
