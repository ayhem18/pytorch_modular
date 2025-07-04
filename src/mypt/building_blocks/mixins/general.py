import torch

from torch import nn
from typing import Iterator, List, Tuple



def generic_to(module: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
    """This is a hacky solution simply because I do not exactly understand 

    Args:
        module (torch.nn.Module): _description_

    Returns:
        torch.nn.Module: _description_
    """
    
    if isinstance(module, nn.Sequential):
        for item in module:
            generic_to(item, *args, **kwargs)
    
    elif isinstance(module, nn.ModuleDict):
        for _, item in module.items():
            generic_to(item, *args, **kwargs)

    elif isinstance(module, nn.ModuleList):
        for i in range(len(module)):
            module[i] = module[i].to(*args, **kwargs)

    return module.to(*args, **kwargs)



class ModuleListMixin:
    """
    This mixin is used to provide default common implementations for a class that uses a ModuleList field.
    Most these methods are expected to be part of more thorough implementation in the child class.
    """
    def __init__(self, inner_model_field_name: str):
        self._inner_model_field_name = inner_model_field_name
        self._verified = False

    def _verify_instance_moduleListMixin(self): 
        if self._verified:
            return

        if not hasattr(self, self._inner_model_field_name):
            raise AttributeError(f"the child class is expected to have the attribute '{self._inner_model_field_name}'")
        
        if not isinstance(getattr(self, self._inner_model_field_name), (torch.nn.ModuleList)):
            raise TypeError(f"The ModuleListMixin expects the self.{self._inner_model_field_name} attribute to be of type {torch.nn.ModuleList}. Found: {type(getattr(self, self._inner_model_field_name))}")

        for module in getattr(self, self._inner_model_field_name):
            if not isinstance(module, nn.Module):
                raise TypeError(f"The ModuleListMixin expects the self.{self._inner_model_field_name} attribute to be a torch.nn.ModuleList of nn.Module instances. Found: {type(module)}")

        self._verified = True


    def module_list_to(self, *args, **kwargs) -> 'ModuleListMixin':
        self._verify_instance_moduleListMixin()

        inner_model = getattr(self, self._inner_model_field_name)

        # call the '.to' method for each Module in the ModuleList
        for i in range(len(inner_model)):
            inner_model[i] = inner_model[i].to(*args, **kwargs)
        
        return self

    def module_list_train(self, mode: bool = True) -> 'ModuleListMixin':
        self._verify_instance_moduleListMixin()

        # make sure to set the training attribute of the module itself !!!
        self.training = mode
 
        inner_model = getattr(self, self._inner_model_field_name)

        for i in range(len(inner_model)):
            inner_model[i].train(mode)

        return self 
    
    def module_list_eval(self) -> 'ModuleListMixin':
        self._verify_instance_moduleListMixin()

        return self.module_list_train(mode=False) 
    

    def module_list_modules(self) -> Iterator[nn.Module]:
        yield self
        
        self._verify_instance_moduleListMixin()

        inner_model = getattr(self, self._inner_model_field_name)

        for i in range(len(inner_model)):
            gen = inner_model[i].modules()
            for module in gen:
                yield module
        
    def module_list_parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        self._verify_instance_moduleListMixin()

        inner_model = getattr(self, self._inner_model_field_name)

        for i in range(len(inner_model)):
            gen = inner_model[i].parameters(recurse)
            for param in gen:
                yield param
    
    def module_list_named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        self._verify_instance_moduleListMixin()

        inner_model = getattr(self, self._inner_model_field_name)

        for i in range(len(inner_model)):
            gen = inner_model[i].named_parameters(prefix, recurse)
            for param_name, param in gen:
                if prefix:
                    yield f"{prefix}.{self._inner_model_field_name}.{i}.{param_name}", param
                else:
                    yield f"{self._inner_model_field_name}[{i}].{param_name}", param




    def __len__(self) -> int:
        self._verify_instance_moduleListMixin()

        inner_model = getattr(self, self._inner_model_field_name)

        return len(inner_model)

    def __getitem__(self, index: int) -> nn.Module:
        self._verify_instance_moduleListMixin()

        inner_model = getattr(self, self._inner_model_field_name)

        if index < 0 or index >= len(inner_model):
            raise IndexError(f"Index {index} is out of bounds for the ModuleListMixin. The ModuleList has {len(inner_model)} modules.")

        return inner_model[index]




class SequentialModuleListMixin(ModuleListMixin):
    """
    This mixin provides an implementation of the forward method for a class that uses a ModuleList but with a sequential structure.
    """
    
    def sequential_module_list_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._verify_instance_moduleListMixin()

        inner_model = getattr(self, self._inner_model_field_name)

        # call the forward method for each Module in the ModuleList
        for i in range(len(inner_model)):
            x = inner_model[i](x, *args, **kwargs)
        return x



class NonSequentialModuleMixin:
    """
    This mixin provides an implementation of the default methods for Modules with different components that play role in the forward pass.
    """
    
    def __init__(self, inner_components_fields: List[str]):
        self._inner_components_fields = inner_components_fields


    def to(self, *args, **kwargs) -> 'NonSequentialModuleMixin':
        for field in self._inner_components_fields:
            field_as_module = getattr(self, field) 

            if field_as_module is None:
                continue

            # use the generic_to method to handle tricky cases such as Sequential, ModuleDict, ModuleList and so on... 
            field_as_module = generic_to(field_as_module, *args, **kwargs)

        return self

    def train(self, mode: bool = True) -> 'NonSequentialModuleMixin':
        self.training = mode

        for field in self._inner_components_fields:
            field_as_module = getattr(self, field) 

            if field_as_module is None:
                continue
            
            field_as_module.train(mode)

        return self
    
    def eval(self) -> 'NonSequentialModuleMixin':
        return self.train(mode=False)
    
    def children(self) -> Iterator[nn.Module]:
        for field in self._inner_components_fields:
            field_as_module = getattr(self, field)

            if field_as_module is None:
                continue

            yield field_as_module

    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        for field in self._inner_components_fields:
            field_as_module = getattr(self, field)

            if field_as_module is None:
                continue

            yield field, field_as_module

    def modules(self) -> Iterator[nn.Module]:
        yield self
        for field in self._inner_components_fields:
            field_as_module = getattr(self, field)

            if field_as_module is None:
                continue

            for m in field_as_module.modules():
                yield m

    def parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        for field in self._inner_components_fields:
            field_as_module = getattr(self, field)

            if field_as_module is None:
                continue

            for p in field_as_module.parameters(recurse):
                yield p

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        for field in self._inner_components_fields:
            field_as_module = getattr(self, field)

            if field_as_module is None:
                continue

            for name, p in field_as_module.named_parameters(prefix, recurse):
                yield f"{field}.{name}", p
