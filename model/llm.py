from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler


class Model_Core:
    def __init__(
        self, 
        model_path, 
        mmproj_path, 
        n_ctx=2048, 
        n_gpu_layers=-1,
    ) -> None:
        
        # ------------------
        # Initializing Model
        # ------------------
        
        # Load Vision Model
        self.chat_handler = Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False,
        )

        # Load LLM
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
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