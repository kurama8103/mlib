# %%
import sys
from PIL import Image


def resize_img(filepath: str, max_size: int = 300) -> Image:
    img = Image.open(filepath)
    img = img.resize(
        [max_size * i // max(img.width, img.height) for i in [img.width, img.height]]
    )
    return img


if __name__ == "__main__":
    if "ipykernel_launcher.py" in sys.argv[0]:
        filepath = "../data/test.jpg"
    else:
        filepath = sys.argv[1]
    img = resize_img(filepath)
    img.show()
# %%
