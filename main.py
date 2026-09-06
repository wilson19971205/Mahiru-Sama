import os
import sys
import pyaudio
import config
from config import *

from utils import *
from module.llm import LLM_Core
from module.stt import STT_Core


def main():

    # ---------------
    # Load Model core
    # ---------------
    print("Loading LLM...")
    llm = LLM_Core(config)
    print("LLM Load complete!")

    # Test image
    image_url = image_to_url(IMAGE_PATH)


    # ---------------
    # Load STT Module
    # ---------------
    print("Loading STT Module...")
    stt = STT_Core(config)
    print("STT Load complete!")

    stt.start_listening()


    # ----------------------------
    # Modules loading all Complete
    # ----------------------------

    while True:
        text = input("User: ")
        
        if text == "exit":
            break

        response = llm.chat(text, image_url)
        print("AI: ", response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
