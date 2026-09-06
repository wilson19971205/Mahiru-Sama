from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler


class LLM_Core:
    def __init__(self, config) -> None:
        
        # ------------
        # Initializing
        # ------------

        self.config = config
        self.model_path = getattr(
            config, 
            "MODEL_PATH", 
            "./module/llm_model/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        )
        self.mmproj_path = getattr(
            config, 
            "MMPROJ_PATH", 
            "./module/llm_model/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        )
        self.n_ctx = getattr(config, "LLM_N_CTX", 2048)
        self.n_gpu_layers = getattr(config, "LLM_N_GPU_LAYERS", -1)
        
        # Load Vision Model
        self.chat_handler = Qwen25VLChatHandler(
            clip_model_path=self.mmproj_path,
            verbose=False,
        )

        # Load LLM
        self.llm = Llama(
            model_path=self.model_path,
            chat_handler=self.chat_handler,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def chat(self, text, image=None):
        if image is None:
            messages = [
                {
                    "role": "user",
                    "content": text
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": text
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image}
                        }
                    ]
                }
            ]

        return self.llm.create_chat_completion(messages=messages)