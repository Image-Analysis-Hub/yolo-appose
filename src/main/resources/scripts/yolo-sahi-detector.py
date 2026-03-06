# YOLO-SAHI dectector

import numpy as np
import tifffile
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

report = print
def listen(callback):
    global report
    report = callback

def detect_sporozoites_yolo(model, image, slice_height=128, slice_width=128, overlap_height_ratio=0.2, overlap_width_ratio=0.2, savefig=False, outdir=None, filename=None, option="both"):

    result = get_sliced_prediction(
        image,
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    if savefig & (outdir is not None):
        if option=="both":
            result.export_visuals(export_dir=outdir,text_size=0.2,rect_th=1,hide_labels=False,file_name=filename+"_withscore")
            result.export_visuals(export_dir=outdir,text_size=0.2,rect_th=1,hide_labels=True,file_name=filename+"_noscore")
        elif option=="score":
            result.export_visuals(export_dir=outdir,text_size=0.2,rect_th=1,hide_labels=False,file_name=filename)
        elif option=="noscore":
            result.export_visuals(export_dir=outdir,text_size=0.2,rect_th=1,hide_labels=True,file_name=filename)

    return result

# MAIN PART
from skimage import io

appose_mode = 'task' in globals()
if appose_mode:
    listen(task.update)
else:
    from appose.python_worker import Task
    task = Task()

def open_img(path_to_img):
    """Returns image reading from tifffile."""
    # return tifffile.imread(path_to_img)
    return io.imread(path_to_img, plugin="tifffile")

def flip_img(img):
    """Flips a NumPy array between Java (F_ORDER) and NumPy-friendly (C_ORDER)"""
    return np.transpose(img, tuple(reversed(range(img.ndim))))

def share_as_ndarray(img):
    """Copies a NumPy array into a same-sized newly allocated block of shared memory"""
    from appose import NDArray
    shared = NDArray(str(img.dtype), img.shape)
    shared.ndarray()[:] = img
    return shared

# Obtain the original image
if appose_mode:
    img = flip_img(img_apos.ndarray()) # img_apos is the input variable from Appose
    task.update(f"Input image of shape: {img.shape}")
else:
    path_to_img = "/data/IAH/DevProjects/yolosahi-fiji/images/sporozoite.tif"
    img = open_img(path_to_img)
    print(img.shape)

# model_path = "/data/IAH/DevProjects/yolosahi-fiji/models/best.pt"
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path_apos, # model_path_apos is the input variable from Appose
    confidence_threshold=confidence_threshold_apos, # this can be choosed according to F1 curve
    # image_size=256, # important set it according to training image size
    device="cuda:0",  # 'cpu' or 'cuda:0'
)

# Run yolo-sahi detection
result = detect_sporozoites_yolo(
    model=detection_model,
    image=img,
    slice_height=slice_height_apos, 
    slice_width=slice_width_apos, 
    overlap_height_ratio=overlap_height_ratio_apos, 
    overlap_width_ratio=overlap_width_ratio_apos
)

# # Get bbox results
# bbox_label = np.zeros((img.shape[0], img.shape[1]), dtype=np.int16)
# lbl_ix = 1
# for item in result.to_coco_annotations():
#     x, y, w, h = item['bbox']
#     x, y, w, h = int(x), int(y), int(w), int(h)
#     bbox_label[y:y+h, x:x+w] = lbl_ix
#     lbl_ix += 1

# Get bbox results as a list of dicts
bboxtbl = []
for item in result.to_coco_annotations():
    x, y, w, h = item['bbox']
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Convert xyxy → (x, y, width, height) for Fiji Roi
    bbox = {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "confidence": float(item['score']),
        "class_id": int(item['category_id']),
        "class_name": item['category_name']
    }
    bboxtbl.append(bbox)  


# # plot bbox label
# import matplotlib.pyplot as plt
# plt.imshow(bbox_label)
# plt.show()

if appose_mode:
    # task.update(f"Output bbox label of shape: {bbox_label.shape}, dtype: {bbox_label.dtype}, nb values: {len(np.unique(bbox_label))-1}")
    # task.outputs["bboxlabel"] = share_as_ndarray(flip_img(bbox_label))
    task.outputs["bboxtable"] = bboxtbl



