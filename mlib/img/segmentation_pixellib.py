# %%
import matplotlib.pyplot as plt
import sys
from pixellib.instance import instance_segmentation
from PIL import Image


def segmentation_pixellib(
    image_path,
    output_image_name='out.jpg',
    model_path='mask_rcnn_coco.h5'
):
    segment_image = instance_segmentation()
    segment_image.load_model(model_path)
    segmask, output = segment_image.segmentImage(
        image_path=image_path,
        show_bboxes=True,
        output_image_name=output_image_name,
        extract_segmented_objects=False,
        save_extracted_objects=False)

    dic_class = [{
        segment_image.model.config.class_names[k]:
        segmask['scores'][i]
    } for i, k in enumerate(segmask['class_ids'])]

    return output_image_name, segmask, output, dic_class


if __name__ == '__main__':
    if 'ipykernel_launcher.py' in sys.argv[0]:
        image_path = '../data/test.jpg'
    else:
        image_path = sys.argv[1]
    f, s, o, d = segmentation_pixellib(image_path)
    plt.imshow(o[:, :, ::-1])
    # image = Image.open(f)
    # image.show()

# %%
