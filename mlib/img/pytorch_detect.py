# %%
import numpy as np
import cv2
import random
import PIL
import matplotlib.pylab as plt
import torch
import torchvision
import sys

# eval('!pip3 install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html')

# imagenet_data = torchvision.io.read_image('../data/test.jpg')
# data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                           batch_size=4,
#                                           shuffle=True)

coco_names = [
    "unlabeled",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "bird",
    "leaf",
]


def detect(img) -> dict:
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.resnet18(pretrained=True)
    model = model.eval()

    image_tensor = torchvision.transforms.functional.to_tensor(img)
    output = model([image_tensor])[0]
    return output


def draw_box(img, output: dict, coco_names: list) -> np.ndarray:
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]
    result_image = np.array(img.copy())

    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score > 0.5:
            color = random.choice(colors)

            # draw box
            # line thickness
            tl = round(0.002 * max(result_image.shape[0:2])) + 1
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(result_image, c1, c2, color, thickness=tl)
            # draw text
            display_txt = "%s: %.1f%%" % (coco_names[label], 100 * score)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(result_image, c1, c2, color, -1)  # filled
            cv2.putText(
                result_image,
                display_txt,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
    return result_image


if __name__ == "__main__":
    if "ipykernel_launcher.py" in sys.argv[0]:
        filepath = "../data/test.jpg"
    else:
        filepath = sys.argv[1]

    img = PIL.Image.open(filepath)
    # plt.imshow(image)
    output = detect(img)

    result_image = draw_box(img, output, coco_names)
    plt.imshow(result_image)

    masks = None
    for score, mask in zip(output["scores"], output["masks"]):
        if score > 0.5:
            if masks is None:
                masks = mask
            else:
                masks = torch.max(masks, mask)

    # plt.imshow(masks.squeeze(0).detach().numpy())
    masks.detach().shape, masks.squeeze(0).detach().shape

# %%
