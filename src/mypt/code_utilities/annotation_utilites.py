"""
This script contains some utility functions to better work with annotations of different tasks (currently on Object Detection annotation functions)
"""
import itertools

from typing import Tuple, List, Union

# let's start with verification
IMG_SHAPE_TYPE = Tuple[int, int]

OBJ_DETECT_ANN_TYPE = List[Union[float, int]]

# the supported formats can be found in the following page of the albumentations documentation:
# https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

COCO = 'coco'
PASCAL_VOC = 'pascal_voc'
YOLO = 'yolo'
ALBUMENTATIONS = 'albumentations'

OBJ_DETECT_ANN_FORMATS = [COCO, PASCAL_VOC, YOLO, ALBUMENTATIONS]


DEFAULT_BBOX_BY_FORMAT = {COCO: [0, 0, 1, 1], YOLO: [0, 0, 0.1, 0.1], PASCAL_VOC: [0, 0, 1, 1], ALBUMENTATIONS: [0.0, 0.0, 0.1, 0.1]}


######################################################## OBJECT DETECTION FORMAT VERIFICATION ########################################################

def verify_object_detection_bbox(annotation) -> OBJ_DETECT_ANN_TYPE:
    # proceed with checking the annotations
    if len(annotation) not in [2, 4]:
        raise ValueError(f"The annotation is expected to be of length 2 or 4. Found: {len(annotation)}")

    if len(annotation) == 2 and not (isinstance(annotation[0], (Tuple, List)) and isinstance(annotation[1], (Tuple, List))):
        raise ValueError(f"found an annotation of size 2 whose elements are not iterables")

    # flatten the annotation 
    flattened_ann = list(itertools.chain(*annotation)) if len(annotation) == 2 else annotation
    if len(flattened_ann) != 4:
        raise ValueError(f"Each bounding box annotation is expected to contain exactly 4 values. Found: {len(flattened_ann)}")
    
    for a in flattened_ann: 
        if not isinstance(a, (float, int)):
            raise ValueError(f"the bounding boxes annotations are expected to be numerical values. Found: {a} of type: {type(a)}")

    return flattened_ann


def _verify_pascal_voc_format(annotation: OBJ_DETECT_ANN_TYPE, 
                             img_shape: IMG_SHAPE_TYPE, 
                             normalize: bool = True) -> OBJ_DETECT_ANN_TYPE:
    
    x_min, y_min, x_max, y_max = annotation
    
    if not all([isinstance(x, int) for x in annotation]):
        raise ValueError(f"the pascal_voc format is not normalized")

    if not (x_min < x_max and x_min >= 0 and x_max <= img_shape[1] and x_max >= 1):
        raise ValueError(f"elements 1 and 3 must represent x_min and x_max")

    if not (y_min < y_max and y_min >= 0 and y_max <= img_shape[0] and y_max >= 1):
        raise ValueError(f"elements 2 and 4 must represent y_min and y_max")
    
    if normalize:
        x_min /= img_shape[1]
        x_max /= img_shape[1]

        y_min /= img_shape[0]
        y_max /= img_shape[0]

    return [x_min, y_min, x_max, y_max]


def _verify_coco_format(annotation: OBJ_DETECT_ANN_TYPE, 
                             img_shape: IMG_SHAPE_TYPE, 
                             normalize: bool = True) -> OBJ_DETECT_ANN_TYPE:
    
    
    if not all([isinstance(x, int) for x in annotation]):
        raise ValueError(f"the pascal_voc format is not normalized")

    # width represents the length of the image on the x-axis
    # height represents the length of the image on the y-axis

    x_min, y_min, w, h = annotation

    if not (img_shape[1] >= x_min >= 0 and img_shape[0] >= y_min >= 0):
        raise ValueError("elements 1 and 2 should represent x_min, y_min respectively")

    if not (img_shape[1] >= w > 0 and img_shape[0] >= h > 0):
        raise ValueError("elements 3 and 4 should represent the width and the height respectively")

    if normalize:
        x_min /= img_shape[1]
        y_min /= img_shape[0]

        w /= img_shape[1]
        h /= img_shape[1]


    return [x_min, y_min, w, h]


def _verify_albumentations_format(annotation: OBJ_DETECT_ANN_TYPE, 
                             img_shape: IMG_SHAPE_TYPE, 
                             normalize: bool = True) -> OBJ_DETECT_ANN_TYPE:
    
    # the normalize argument was not removed just to have a uniform function signature for all supported formats 
    if not normalize:
        raise ValueError(f"The normalize argument must be set to True since it is at the core of the format !!")

    x_min, y_min, x_max, y_max = annotation
    
    if not all([isinstance(x, float) and 1 >= x >= 0 for x in annotation]):
        raise ValueError(f"the albumentations format is supposed to be normalized")
    
    if not (x_min < x_max and x_min >= 0):
        raise ValueError(f"elements 1 and 3 must represent x_min and x_max")

    if not (y_min < y_max and y_min >= 0):
        raise ValueError(f"elements 2 and 4 must represent y_min and y_max")

    return [x_min, y_min, x_max, y_max]


def _verify_yolo_format(annotation: OBJ_DETECT_ANN_TYPE, 
                        img_shape: IMG_SHAPE_TYPE, 
                        normalize: bool = True) -> OBJ_DETECT_ANN_TYPE:
    
    # the normalize argument was not removed just to have a uniform function signature for all supported formats 
    if not normalize:
        raise ValueError(f"The normalize argument must be set to True since it is at the core of the format !!")

    x_center, y_center, w_n, h_n = annotation
    
    if not all([isinstance(x, float) and 1 >= x >= 0 for x in annotation]):
        raise ValueError(f"the albumentations format is supposed to be normalized")
    
    if x_center < w_n / 2:
        raise ValueError("The x_center must be larger or equal to half the width")

    if (x_center + w_n / 2) > 1:
        raise ValueError(f"the sum of the x_center and half the width exceed 1 !!!")

    if y_center < h_n / 2:
        raise ValueError("The y_center must be larger or equal to half the height")

    if (y_center + h_n / 2) > 1:
        raise ValueError(f"the sum of the y_center and half the height exceed 1 !!!")

    return annotation


__ann_verification_dict = {'coco': _verify_coco_format, 'pascal_voc': _verify_pascal_voc_format, 'yolo': _verify_yolo_format, 'albumentations': _verify_albumentations_format}


def verify_object_detection_ann_format(annotation: OBJ_DETECT_ANN_TYPE, 
                                   current_format: str, 
                                   img_shape: IMG_SHAPE_TYPE,
                                   normalize: bool=True) -> OBJ_DETECT_ANN_TYPE:
    if current_format.lower() not in OBJ_DETECT_ANN_FORMATS:
        raise NotImplementedError(f"The current format: {current_format} is not supported")
    return __ann_verification_dict[current_format](annotation=annotation, img_shape=img_shape, normalize=normalize)


######################################################## OBJECT DETECTION FORMAT CONVERSION ########################################################

################################ 2 COCO ################################

def _pascal_voc_2_coco(annotation: OBJ_DETECT_ANN_TYPE, 
                        img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    x_min, y_min, x_max, y_max = annotation
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def _yolo_2_coco(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    
    x_cn, y_cn, w_n, h_n = annotation

    # normalize width and height 
    w, h = int(round(w_n * img_shape[1])), int(round(h_n * img_shape[0]))

    x_min = int(round((x_cn - w_n / 2) * img_shape[1]))
    y_min = int(round((y_cn - h_n / 2) * img_shape[0]))

    return [x_min, y_min, w, h]

def _albumentations_2_coco(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    
    x_min_n, y_min_n, x_max_n, y_max_n = annotation

    # scale
    x_min, x_max = int(round(x_min_n * img_shape[1])), int(round(x_max_n * img_shape[1]))
    y_min, y_max = int(round(y_min_n * img_shape[0])), int(round(y_max_n * img_shape[0]))

    return [x_min, y_min, x_max - x_min, y_max - y_min]


################################ 2 YOLO ################################

def _pascal_voc_2_yolo(annotation: OBJ_DETECT_ANN_TYPE, 
                        img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    x_min, y_min, x_max, y_max = annotation
    # calculate the center, width and height
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_max + y_min) / 2
    # normalize
    res = [x_center / img_shape[1], y_center / img_shape[0], width / img_shape[1], height / img_shape[0]]
    return res

def _coco_2_yolo(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    x_min, y_min, width, height = annotation
    x_center, y_center = x_min + width / 2, y_min + height / 2
    res = [x_center / img_shape[1], y_center / img_shape[0], width / img_shape[1], height / img_shape[0]]
    return res

def _albumentations_2_yolo(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    x_min_n, y_min_n, x_max_n, y_max_n = annotation
    x_center, y_center = (x_min_n + x_max_n) / 2, (y_min_n + y_max_n) / 2 

    return [x_center, y_center, x_max_n - x_min_n, y_max_n - y_min_n]

################################ 2 Pascal_voc ################################
def _yolo_2_pascal_voc(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    
    x_cn, y_cn, w_n, h_n = annotation

    # normalize width and height 
    w, h = int(round(w_n * img_shape[1])), int(round(h_n * img_shape[0]))

    x_min = int(round((x_cn - w_n / 2) * img_shape[1]))
    y_min = int(round((y_cn - h_n / 2) * img_shape[0]))

    res = [x_min, y_min, x_min + w, y_min + h]
    return res

def _albumentations_2_pascal_voc(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    x_min_n, y_min_n, x_max_n, y_max_n = annotation
    res = [int(round(x_min_n * img_shape[1])), int(round(y_min_n * img_shape[0])), int(round(x_max_n * img_shape[1])), int(round(y_max_n  * img_shape[0]))]
    return res

def _coco_2_pascal_voc(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    x_min, y_min, width, height = annotation
    return [x_min, y_min, x_min + width, y_min + height]

################################ 2 albumentations ################################

def _yolo_2_albumentations(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    
    x_cn, y_cn, w_n, h_n = annotation
    res = [round(x_cn - w_n / 2, 4), round(y_cn - h_n / 2, 4), round(x_cn + w_n / 2, 4),  round(y_cn + h_n / 2, 4)]
    return res

def _coco_2_albumentations(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    x_min, y_min, width, height = annotation
    x_max, y_max = x_min + width, y_min + height
    res = [round(x_min / img_shape[1], 4), round(y_min / img_shape[0], 4), round(x_max / img_shape[1], 4), round(y_max / img_shape[0],4)]
    return res 

def _pascal_voc_2_albumentations(annotation: OBJ_DETECT_ANN_TYPE, 
                img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:    
    x_min, y_min, x_max, y_max = annotation
    res = [round(x_min / img_shape[1], 4), round(y_min / img_shape[0], 4), round(x_max / img_shape[1], 4), round(y_max / img_shape[0], 4)]
    return res 


def convert_bbox_annotation(annotation: OBJ_DETECT_ANN_TYPE, current_format: str, target_format: str, img_shape: IMG_SHAPE_TYPE) -> OBJ_DETECT_ANN_TYPE:
    if current_format not in OBJ_DETECT_ANN_FORMATS or target_format not in OBJ_DETECT_ANN_FORMATS:
        raise NotImplementedError(f"currently supporting only the following formats: {OBJ_DETECT_ANN_FORMATS}")

    # definitely a bad practice...
    return eval(f'_{current_format}_2_{target_format}')(annotation=annotation, img_shape=img_shape)
