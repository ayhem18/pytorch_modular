"""
This script contains another abstract class of the 'abstract_classifier_wrapper' class; it is an abstract implement for wrappers with known backbones: resnet and alexnet
"""

import os, torch
# from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, List, Any

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

from .abstract_classifier_wrapper import ClassifierWrapper


class BackboneClassifierWrapper(ClassifierWrapper):
    def __init__(self, 
                input_shape: Tuple,
                num_classes: int,   

                optimizer_class: callable,
                learning_rate: Optional[float] = 10 ** -4,
                num_vis_images: int = 3,
                scheduler_class: Optional[callable] = None,
                optimizer_keyargs: Optional[Dict] = None,
                scheduler_keyargs: Optional[Dict] = None,
                ):
        # parent class constructor
        super().__init__(
                        input_shape=input_shape, 
                        num_classes=num_classes, 

                        optimizer_class=optimizer_class,
                        learning_rate=learning_rate,
                        scheduler_class=scheduler_class,
                        
                        num_vis_images=num_vis_images,

                        optimizer_keyargs=optimizer_keyargs,
                        scheduler_keyargs=scheduler_keyargs                
                        )

        # if either self.scheduler_args or self.scheduler_class are lists: basically different rl schedulers for the backbone and
        # the classification head, then we will have 2 optimizers for each component and we need to activate self.automatica_optimizer
        if isinstance(self.scheduler_args, List) or isinstance(self.scheduler_class, List):
            self.automatic_optimization = False

        
        # the child class must implement these concrete fields
        self.fe, self.flatten_layer, self.classification_head, self._model = None, None, None, None 


    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        if self.automatic_optimization:
            return super().training_step(batch, batch_idx, *args, **kwargs)

        # forward pass
        _, _, loss_obj, accuracy =  self._forward_pass(batch)

        self.log_dict({"train_cls_loss": loss_obj.cpu().item(),
                       "train_accuracy": round(accuracy, 5)})

        # extract the optimizers
        opt1, opt2 = self.optimizers()
        # set each of them to zero grad
        opt1.zero_grad()
        opt2.zero_grad()

        self.manual_backward(loss_obj)

        opt1.step()
        opt2.step()

    def configure_optimizers_multiple_lrs(self) -> Union[torch.optim.Optimizer, 
                                                         Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]], 
                                                         Dict
                                                         ]:
        # this method assumes self.lr is a list: multiple learning rates are passed
        if not (isinstance(self.lr, List) and len(self.lr) == 2):
            raise ValueError(f"the method 'configure_optimizers_multiple_lrs must be called only when 2 learning rates are passed")

        # extract the learning rates 
        l1, l2 = self.lr
        # first build 2 parameter groups
        p1 = [{"params": self.fe.parameters(), "lr": l1}, # set the backbone with the first learning rate
                {"params": self.flatten_layer.parameters(), "lr": l1},  # set the flatten layer with the first learning rate
            ]
        p2 = [{"params": self.classification_head.parameters(), "lr": l2} # set the classification head with the 2nd learning rate
                ]

        if self.scheduler_class is None:
            # create the final set of parameters by concatenating p1 and p2
            optimizer = self.opt(p1 + p2) if (self.opt_args is None) else (self.opt(p1 + p2, **self.opt_args)) 
            return optimizer            

        # we proceed depending on the values for self.scheduler_args and self.scheduler_class
        # if only one class and one set of keyargs are passed, then we return only one optimizer and make use of automatic optimization
        if not (isinstance(self.scheduler_args, List) or isinstance(self.scheduler_class, List)):
            # initialize the optimizer
            optimizer = self.opt(p1 + p2) if (self.opt_args is None) else (self.opt(p1 + p2, **self.opt_args))             
            scheduler = self.scheduler_class(optimizer=optimizer, **self.scheduler_args)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        # otherwise the learning rate in each component is updated differently: we need 2 optimizers
        opt1 = self.opt(p1) if self.opt_args is None else self.opt(p1, **self.opt_args)
        opt2 = self.opt(p2) if self.opt_args is None else self.opt(p2, **self.opt_args)

        # at this point, either self.scheduler_args or self.scheduler_class is a list (or both ...)
        c1, c2 = self.scheduler_class if isinstance(self.scheduler_class, List) else (self.scheduler_class, self.scheduler_class)
        k1, k2 = self.scheduler_args if isinstance(self.scheduler_args, List) else (self.scheduler_args, self.scheduler_args)
        
        ss = [c1(optimizer=opt1, **k1), c2(optimizer=opt2, **k2)]
        return [opt1, opt2], ss 

    # override the 'configure_optimizers' method to allow having different learning rate schedulers for different parts of of the classifier
    def configure_optimizers(self):
        if isinstance(self.lr, List) or isinstance(self.scheduler_args, List) or isinstance(self.scheduler_class, List):
            # first check if we have different learning rates 
            if isinstance(self.lr, List):
                return self.configure_optimizers_multiple_lrs()                

            # at this point we know that self.lr is only one value, but we would like to optimizer            
            p1 = [{"params": self._model.feature_extractor.parameters(), "lr": self.lr}, 
                  {"params": self._model.flatten_layer.parameters(), "lr": self.lr},  
                ]

            p2 = [{"params": self._model.concept_projection.parmaeters(), "lr": self.lr}, 
                  {"params": self._model.classification_head.parameters(), "lr": self.lr} 
                  ]

            opt1 = self.opt(p1) if self.opt_args is None else self.opt(p1, **self.opt_args)
            opt2 = self.opt(p2) if self.opt_args is None else self.opt(p2, **self.opt_args)

            c1, c2 = self.scheduler_class if isinstance(self.scheduler_class, List) else (self.scheduler_class, self.scheduler_class)
            k1, k2 = self.scheduler_args if isinstance(self.scheduler_args, List) else (self.scheduler_args, self.scheduler_args)
            
            ss = [c1(optimizer=opt1, **k1), c2(optimizer=opt2, **k2)]
            return [opt1, opt2], ss 

        # if neither self.lr, self.scheduler_args not self.scheduler_class is a list, then we can simply call the parent method
        return super().configure_optimizers()

    # override the 'configure_optimizers' method to allow having different optimizers for different parts of of the classifier
    def configure_optimizers(self):
        if isinstance(self.lr, List) or isinstance(self.scheduler_args, List) or isinstance(self.scheduler_class, List):
            # first check if we have different learning rates 
            if isinstance(self.lr, List):
                return self.configure_optimizers_multiple_lrs()                

            # at this point we know that self.lr is only one value, but we would like to optimizer            
            p1 = [{"params": self.fe.parameters(), "lr": self.lr}, # set the backbone with the first learning rate
                  {"params": self.flatten_layer.parameters(), "lr": self.lr},  # set the flatten layer with the first learning rate
                ]

            p2 = [{"params": self.classification_head.parameters(), "lr": self.lr} # set the classification head with the 2nd learning rate
                  ]

            opt1 = self.opt(p1) if self.opt_args is None else self.opt(p1, **self.opt_args)
            opt2 = self.opt(p2) if self.opt_args is None else self.opt(p2, **self.opt_args)

            c1, c2 = self.scheduler_class if isinstance(self.scheduler_class, List) else (self.scheduler_class, self.scheduler_class)
            k1, k2 = self.scheduler_args if isinstance(self.scheduler_args, List) else (self.scheduler_args, self.scheduler_args)
            
            ss = [c1(optimizer=opt1, **k1), c2(optimizer=opt2, **k2)]
            return [opt1, opt2], ss 

        # if neither self.lr, self.scheduler_args not self.scheduler_class is a list, then we can simply call the parent method
        return super().configure_optimizers() ## don't forget the "return" statement
