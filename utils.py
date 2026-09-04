import base64

# Encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Image to url
def image_to_url(image_path):
    return f"data:image/jpeg;base64,{encode_image(image_path)}"
