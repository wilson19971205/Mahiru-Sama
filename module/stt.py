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
import torch  # Silero 內部處理 Tensor 需要
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from silero_vad import load_silero_vad  # 僅載入核心模型

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class STT_Core:

    def __init__(self, config) -> None:
        self.config = config

        self.model_path = getattr(config, "STT_MODEL_PATH", "./module/stt_model/Qwen3-ASR-0.6B-bf16.gguf")
        self.mmproj_path = getattr(config, "STT_MMPROJ_PATH", "./module/stt_model/mmproj-Qwen3-ASR-0.6B-bf16.gguf")
        self.n_ctx = getattr(config, "STT_N_CTX", 2048)
        self.n_gpu_layers = getattr(config, "STT_N_GPU_LAYERS", -1)
        self.rate = getattr(config, "STT_SAMPLE_RATE", 16000)

        lang_format = getattr(config, "LANGUAGE_FORMAT", "s2twp")
        self.cc = opencc.OpenCC(lang_format)

        # 1. 🛠️ 初始化 Silero VAD 模型
        print("🔄 [VAD] 正在載入 Silero VAD 模型...")
        self.vad_model = load_silero_vad()
        
        # 2. 🛠️ 狀態機控制變數
        # Silero VAD 官方最推薦的流式區塊大小為 512 個採樣點 (16000Hz 下約 32ms)
        self.chunk = 512  
        self.speech_frames = []  
        self.is_speaking = False  
        self.silence_counter = 0  
        
        # 靈敏度設定 (0.0 到 1.0 之間，越低越靈敏，預設 0.5 適合防雜音)
        self.vad_threshold = getattr(config, "VAD_THRESHOLD", 0.5)
        
        # 連續靜音達到 18 個影格 (18 * 32ms = 576ms，約半秒) 判定說完話
        self.max_silence_frames = getattr(config, "VAD_MAX_SILENCE_FRAMES", 18) 
        self.ready_audio_queue = collections.deque()

        print("🔄 [STT] 正在常駐載入 Qwen3-ASR GGUF 核心...")
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

        self.p = pyaudio.PyAudio()
        self.channels = 1
        self.stream = None
        self.is_listening = False

        print("✅ [STT] GGUF 語音與 Silero VAD 核心啟動成功！")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """內部麥克風收音監聽 + Silero 原始模型級別狀態機"""
        if self.is_listening:
            # 轉換為 numpy int16 複本
            audio_int16 = np.frombuffer(in_data, dtype=np.int16).copy()
            
            # 將資料歸一化至 -1.0 ~ 1.0 的 float32，並轉為 PyTorch Tensor
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32)
            
            try:
                # 直接調用模型獲取當前影格的人聲信心值 (0.0 ~ 1.0)
                # 這是流式音訊（Streaming）最安全、絕對不會失效的調用方式
                with torch.no_grad():
                    confidence = self.vad_model(audio_tensor, self.rate).item()
                
                # 判定當前影格是否為人聲
                is_speech = confidence >= self.vad_threshold
            except Exception:
                is_speech = False

            if is_speech:
                if not self.is_speaking:
                    print("\n🎙️  [VAD] 檢測到人聲，開始錄音...", end="", flush=True)
                    self.is_speaking = True
                
                self.speech_frames.append(audio_int16)
                self.silence_counter = 0  
            else:
                if self.is_speaking:
                    # 即使是靜音影格，在說話狀態中也必須保留，避免斷字
                    self.speech_frames.append(audio_int16)
                    self.silence_counter += 1
                    
                    # 超過靜音影格閾值，切斷並觸發 ASR
                    if self.silence_counter > self.max_silence_frames:
                        print(" ⏹️ [VAD] 說話結束，送交 ASR 識別。")
                        full_audio = np.concatenate(self.speech_frames)
                        self.ready_audio_queue.append(full_audio)
                        
                        # 重置狀態機
                        self.is_speaking = False
                        self.speech_frames = []
                        self.silence_counter = 0

        return (None, pyaudio.paContinue)

    def start_listening(self):
        """開啟即時監聽與語音辨識循環"""
        if self.is_listening:
            return

        self.ready_audio_queue.clear()
        self.speech_frames = []
        self.is_speaking = False
        self.silence_counter = 0
        self.is_listening = True

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._audio_callback,
        )

        print("\n🎤 【Qwen3-ASR + Silero VAD】已開啟智慧即時監聽... (按下 Ctrl+C 結束測試)")
        self.stream.start_stream()

        try:
            while self.stream.is_active() and self.is_listening:
                time.sleep(0.05)  

                if len(self.ready_audio_queue) == 0:
                    continue

                full_audio = self.ready_audio_queue.popleft()

                # 音量安全檢查（防止背景絕對靜音時的無意義觸發）
                rms = np.sqrt(np.mean(full_audio.astype(np.float32) ** 2))
                if rms < 80.0: 
                    continue

                try:
                    wav_io = io.BytesIO()
                    wavfile.write(wav_io, self.rate, full_audio)
                    wav_bytes = wav_io.getvalue()

                    audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
                    data_url = f"data:application/octet-stream;base64,{audio_base64}"

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "<|audio_bos|><|AUDIO|><|audio_eos|>",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                },
                            ],
                        }
                    ]

                     # 4. 呼叫建立對話 completion
                    response = self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=64,
                        temperature=0.0,
                    )

                    # 5. ❗ 移除舊的分割邏輯，直接抓取並印出模型完整的文字輸出
                    raw_text = response["choices"][0]["message"]["content"].strip()
                    tw_text = self.cc.convert(raw_text).split("_Solution<asr_text>")[-1].strip()
                    
                    print(f"\n💬 語音辨識結果: {tw_text}\n")

                except Exception:
                    pass

        except KeyboardInterrupt:
            self.stop_listening()

    def stop_listening(self):
        if not self.is_listening:
            return
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("\n🛑 [STT] 語音監聽已關閉。")

    def close(self):
        self.stop_listening()
        self.p.terminate()
        # 釋放 Silero 內部狀態
        if hasattr(self, 'vad_model') and hasattr(self.vad_model, 'reset_states'):
            self.vad_model.reset_states()
