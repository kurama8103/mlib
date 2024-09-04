# %%
import sys
from PIL import Image
from rembg import remove
from resize_img import resize_img


def remove_by_rembg(filepath: str):
    img = resize_img(filepath)
    return remove(img)


if __name__ == "__main__":
    if "ipykernel_launcher.py" in sys.argv[0]:
        filepath = "../data/test.jpg"
    else:
        filepath = sys.argv[1]
    remove_by_rembg(filepath).show()

# %%
