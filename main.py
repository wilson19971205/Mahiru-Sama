import os
import sys

from config import *
from utils import *
from model.llm import Model_Core


def main():

    # Load Model core
    print("Loading model...")
    model = Model_Core(
        model_path=MODEL_PATH,
        mmproj_path=MMPROJ_PATH,
    )
    print("Model Load complete!")

    # Test image
    image_url = image_to_url(IMAGE_PATH)

    while True:
        text = input("User: ")
        
        if text == "exit":
            break

        response = model.chat(text, image_url)
        print("AI: ", response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
