#%%
import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union

#%%
class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]
    
Arrayable=Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable)->np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)
    


#%%
class Tensor:
    def __init__(self, data:Arrayable, requires_grad:bool=False, depends_on:List[Dependency]=None) -> None:
        self.data=ensure_array(data)
        self.requries_grad=requires_grad
        self.depends_on=depends_on or []
        self.shape=self.data.shape
        self.grad: Optional[Tensor] = None
        if self.requries_grad:
            self.zero_grad()
        

    def __repr__(self) -> str:
        return f"Tensor {self.data}, requries_grad = {self.requries_grad}"
    
    def zero_grad(self)->None:
        self.grad=Tensor(np.zeros_like(self.data))
    
    def backward(self, grad:'Tensor'= None):
        assert self.requries_grad, f"Called backward on non-requires-grad tensor"
        
        if grad is None:
            if self.shape==():
                grad=Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")
        self.grad.data+=grad.data
        
        for dependency in self.depends_on:
            backward_grad=dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))
    
    def sum(self)-> 'Tensor':
        return tensor_sum(self)
    
def tensor_sum(t: Tensor) -> Tensor:
    sum_data=t.data.sum()
    requires_grad =t.requries_grad
    if requires_grad == True:
        def grad_fn(grad:np.ndarray) ->np.ndarray:
            return grad*np.ones_like(t.data)       
        depends_on=[Dependency(t, grad_fn)]
    else:
        depends_on=[]
    return Tensor(sum_data, requires_grad, depends_on)

def add(t1:Tensor, t2:Tensor) ->Tensor:
    data=t1.data+t2.data
    requires_grad=t1.requries_grad or t2.requries_grad
    depends_on:List[Dependency]=[]
    if t1.requries_grad==True:
        def grad_fn1(grad:np.ndarray) -> np.ndarray:
            ndims_added=grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad=grad.sum(axis=0)
            return grad
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requries_grad==True:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            ndims_added=grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad=grad.sum(axis=0)
            return grad
        depends_on.append(Dependency(t2, grad_fn2))
        
    return Tensor(data, requires_grad, depends_on)