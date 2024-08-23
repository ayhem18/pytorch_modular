"""
This script contains an implementation of a dataset class designed for the object Detection task.

Such class will be designed after building a clear idea on the object detection task in general
"""

from pathlib import Path
from typing import Union, Optional, Dict, List

from .abstract_ds import ObjectDataset

class ObjectDetectionDs(ObjectDataset):

    def __init__(self,
                 root_dir: Union[str, Path],

                 img_annotations: Optional[Dict],
                 img2ann_dict: Optional[Dict], 
                 read_ann: Optional[callable],
                 
                 target_format: str,
                 current_format: Optional[str],
                 convert: Optional[callable]=None,

                 add_background_label:bool=False,
                 image_extensions: Optional[List[str]]=None
                ) -> None:

        # init the parent class
        super().__init__(root_dir=root_dir,
                    img_annotations=img_annotations,
                    img2ann_dict=img2ann_dict, 
                    read_ann=read_ann,
                    
                    target_format=target_format,
                    current_format=current_format,
                    convert=convert,

                    add_background_label=add_background_label,
                    image_extensions=image_extensions
                    )

        


