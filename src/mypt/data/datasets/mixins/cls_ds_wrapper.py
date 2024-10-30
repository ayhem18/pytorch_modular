"""
This script contains a number of mixins to extend the functionalities of standard classification datasets
"""
import warnings

from typing import Dict, Tuple
from tqdm import tqdm
from bisect import bisect

class ClassificationDsWrapper:
    # a class attribute saving the attributes that child classes are expected to have
    __cls_attrs = ['_ds', 'samples_per_cls_map']

    def _verify_attrs(self):
        # verify the presence of all attributes
        for a in self.__cls_attrs:
            if not hasattr(self, a):
                raise AttributeError(f"Classes inheriting from the `ClassificationDSWrapper` class must have the attribute: {a}")

        # make sure the '_ds' attribute represents a classification dataset
        if len(self._ds) == 0:
            raise ValueError(f"The self._ds must have a positive length. Found: {len(self._ds)}")
        
        if not (isinstance(self._ds[0], Tuple) and len(self._ds[0]) == 2):
            raise ValueError(f"The self._ds attribute is expected to represent a classification dataset; each item represents a tuple (image, class label)")

        if not isinstance(self.samples_per_cls_map, Dict):
            raise TypeError(f"The self.samples_per_cls_map attribute is expected to be a dict. Found: {type(self.samples_per_cls_map)}")


    def _set_samples_per_cls(self, 
                             samples_per_cls: int, 
                             warning: bool = True) -> Dict:

        # first check the attributes
        self._verify_attrs()

        if warning:
            warnings.warn("The `_set_samples_per_cls` expects consecutive samples to be of the same class. Make sure this assumption is satisfied. Otherwise the results would be erroneous")

        # iterate through the dataset
        current_cls = None
        
        last_pointer = 0
        cls_count = 0

        mapping = {0: 0}
        
        for i in tqdm(range(len(self._ds)), desc="iterating through the dataset to set the samples for each class"):
            _, c = self._ds[i]
            
            if current_cls is None:
                current_cls = c
            
            if current_cls == c:
                cls_count += 1    

                if cls_count == samples_per_cls:
                    last_pointer = last_pointer + cls_count
                    

            else:
                if cls_count <= samples_per_cls:
                    last_pointer = last_pointer + cls_count

                mapping[last_pointer] = i 

                cls_count = 1              
                current_cls = c          

        return mapping


    def _find_final_index(self, index: int) -> int:
        self._verify_attrs()
        
        # extract the indices of first samples of each class in the wrapper
        cls_first_sample_indices = sorted(list(self.samples_per_cls_map.keys()))

        index_in_first_sample_list = bisect(cls_first_sample_indices, index) - 1

        cls_index_first_sample_wrapper = cls_first_sample_indices[index_in_first_sample_list]

        cls_index_first_sample_original = self.samples_per_cls_map[cls_index_first_sample_wrapper]

        final_index = cls_index_first_sample_original + (index - cls_index_first_sample_wrapper)

        return final_index

