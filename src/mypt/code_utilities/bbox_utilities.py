"""
This script contains some utility functions to better work with annotations of different tasks (currently on Object Detection annotation functions)
"""
import itertools

from typing import Callable, Iterable, Optional, Tuple, List, Union

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
        raise ValueError(f"elements 1 and 3 must represent x_min and x_max. Found: x_min: {x_min}, x_max: {x_max}. The image dimensions are: {img_shape}")

    if not (y_min < y_max and y_min >= 0 and y_max <= img_shape[0] and y_max >= 1):
        raise ValueError(f"elements 2 and 4 must represent y_min and y_max. Found: y_min: {y_min}, y_max: {y_max}. The image dimensions are: {img_shape}")
    
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

    # scale width and height 
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

    if current_format == target_format:
        # if there is no conversion, make sure the passed annotation / bbox are in the current format
        # set the normalize argument to False
        verify_object_detection_ann_format(annotation=annotation, current_format=current_format, img_shape=img_shape, normalize=False)
        return annotation

    # definitely a bad practice...
    return eval(f'_{current_format}_2_{target_format}')(annotation=annotation, img_shape=img_shape)



######################################################## OBJECT DETECTION BOX GEOMETRIC PROPERTIES ########################################################

def calculate_bbox_area(bbox: OBJ_DETECT_ANN_TYPE, current_format: str, img_shape: Optional[Tuple[int, int]] = None) -> int:
    # convert to the pascal_voc
    if current_format != PASCAL_VOC and img_shape is None:
        raise TypeError(f"calculating the area of a bounding box in a format different from {PASCAL_VOC} requires passing the `img_shape` argument.")
    
    b = convert_bbox_annotation(annotation=bbox, current_format=current_format, target_format=PASCAL_VOC, img_shape=img_shape)

    min_x, min_y, max_x, max_y = b

    return (max_x - min_x + 1) * (max_y - min_y + 1)


def box_within_box(bbox1: OBJ_DETECT_ANN_TYPE, bbox2: OBJ_DETECT_ANN_TYPE, 
                    bbox1_format: str, bbox2_format: str, 
                    img_shape: Optional[Tuple[int, int]] = None) -> int:

    # convert to the pascal_voc
    if (bbox1_format != PASCAL_VOC or bbox2_format != PASCAL_VOC) and img_shape is None:
        raise TypeError(f"calculating the area of a bounding box in a format different from {PASCAL_VOC} requires passing the `img_shape` argument.")

    b1 = convert_bbox_annotation(bbox1, current_format=bbox1_format, target_format=PASCAL_VOC, img_shape=img_shape)
    b2 = convert_bbox_annotation(bbox2, current_format=bbox2_format, target_format=PASCAL_VOC, img_shape=img_shape)

    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2

    # first case b1 in b2
    if x1_min >= x2_min and x1_max <= x2_max and y1_min >= y2_min and y1_min <= y2_max:
        return True 
    
    # second case b2 in b1
    if x2_min >= x1_min and x2_max <= x1_max and y2_min >= y1_min and y2_min <= y1_max:
        return True
    
    return False


def inner_box_wr_outer_box(outer_box: OBJ_DETECT_ANN_TYPE, inner_box: OBJ_DETECT_ANN_TYPE, outer_format: str, inner_format: str, img_shape: Optional[Tuple[int, int]] = None) -> OBJ_DETECT_ANN_TYPE: 
    if not box_within_box(outer_box, inner_box, outer_format, inner_format, img_shape):
        raise ValueError("The inner_box_wr_outer_box function expects to have one box inside the other...")

    b1 = convert_bbox_annotation(outer_box, current_format=outer_format, target_format=PASCAL_VOC, img_shape=img_shape)
    b2 = convert_bbox_annotation(outer_format, current_format=inner_format, target_format=PASCAL_VOC, img_shape=img_shape)

    x1_min, y1_min, _, _ = b1
    x2_min, y2_min, x2_max, y2_max = b2

    # b2 in b1:  x2_min >= x1_min and x2_max <= x1_max and y2_min >= y1_min and y2_min <= y1_max:
    return [x2_min - x1_min, y2_min - y1_min, x2_max - x1_min, y2_max - y1_min]


# some utility functions: whether two bounding boxes interesct
def bounding_boxes_intersect(bbox1: OBJ_DETECT_ANN_TYPE, bbox2: OBJ_DETECT_ANN_TYPE, 
                             img_shape: Tuple[int, int],
                             bbox1_format: str, bbox2_format: str) -> Optional[OBJ_DETECT_ANN_TYPE]:

    # the idea here is simple, convert box bounding boxes to "pascal_voc"
    b1 = convert_bbox_annotation(bbox1, current_format=bbox1_format, target_format=PASCAL_VOC, img_shape=img_shape)
    b2 = convert_bbox_annotation(bbox2, current_format=bbox2_format, target_format=PASCAL_VOC, img_shape=img_shape)

    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2

    min_x_intersection = max(x1_min, x2_min)
    max_x_intersection = min(x1_max, x2_max)

    min_y_intersection = max(y1_min, y2_min)
    max_y_intersection = min(y1_max, y2_max)

    if min_x_intersection < max_x_intersection and min_y_intersection < max_y_intersection:
        return min_x_intersection, min_y_intersection, max_x_intersection, max_y_intersection
    
    return None


def extract_unique_bounding_box(org_bbox: OBJ_DETECT_ANN_TYPE, intersection_bbox: OBJ_DETECT_ANN_TYPE, remove_by: str) -> OBJ_DETECT_ANN_TYPE:
    _verify_pascal_voc_format(org_bbox,normalize=False)
    _verify_pascal_voc_format(intersection_bbox, normalize=False)
    # make sure the intersection_bbox is indeed an intersection bbox
    x1_min, y1_min, x1_max, y1_max = org_bbox
    x2_min, y2_min, x2_max, y2_max = intersection_bbox
    
    if len(set([x1_min, x2_min, x1_max, x2_max])) == 4:
        raise ValueError(f"Make sure the two boxes do indeed intersect !!!. Found 4 different values for the x-coordinates")

    if len(set([y1_min, y2_min, y1_max, y2_max])) == 4:
        raise ValueError(f"Make sure the two boxes do indeed intersect !!!. Found 4 different values for the y-coordinates")

    if remove_by.lower() not in ['x', 'y', 'area']:
        raise NotImplementedError(f"the remove_by argument is expected to be one of the following args: {['x', 'y', 'area']}. Found: {remove_by}")

    if remove_by.lower() == 'x':
        if x1_min == x2_min:
            return x2_max, y1_min, x1_max, y1_max

        if x2_max == x2_max:
            return x1_min, y1_min, x2_min, y1_max

    if remove_by.lower() == "y":
        if y1_min == y2_min:
            return x1_min, y2_min, x1_max, y1_max

        if y2_max == y2_max:
            return x1_min, y1_min, x1_max, y2_min


    # reaching this point means remove_by was set to "area"
    nb1, nb2 = extract_unique_bounding_box(org_bbox, intersection_bbox, remove_by='x'), extract_unique_bounding_box(org_bbox, intersection_bbox, remove_by='y')

    # return the bounding box with the largest area    
    return max([nb1, nb2], key=lambda x: calculate_bbox_area(x, current_format=PASCAL_VOC, img_shape=None))


def extract_contour_bounding_box(contour: Iterable[Tuple[int, int]], 
                                 process_function: Optional[Callable] = None,
                                 y_x: bool = True) -> OBJ_DETECT_ANN_TYPE:
    """a function to extract the bounding box out of a given set of points. The points are assumed connected, however not necessarily representing
    a 'countour' in the sense of the opencv contour. Each point is expected to be (y, x) by default

    Args:
        contour (Iterable[Tuple[int, int]]): a set of connected point

    Returns:
        OBJ_DETECT_ANN_TYPE: The bounding box with the PASCAL format
    """

    if process_function is None:
        # the default function is the identity
        process_function = lambda x: x

    max_x, max_y, min_x, min_y = None, None, None, None

    for element in contour:
        y, x = process_function(element)
        if not y_x:
            # make sure to swap them...
            y, x = x, y

        max_x = max(max_x, x) if max_x is not None else x
        min_x = min(min_x, x) if min_x is not None else x

        max_y = max(max_y, y) if max_y is not None else y
        min_y = min(min_y, y) if min_y is not None else y

    return min_x, min_y, max_x, max_y    