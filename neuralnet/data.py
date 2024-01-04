from typing import Any, Iterator, NamedTuple
import numpy as np

Batch=NamedTuple("Batch",[("inputs", np.ndarray), ("targets", np.ndarray)])

class DataIterator:
    def __Call__(self, inputs: np.ndarray, targets:np.ndarray) -> Iterator[Batch]:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, batch_size:int=32, shuffle:bool=True) -> None:
        self.batch_size=batch_size
        self.shuffle=shuffle
    
    def __call__(self, inputs:np.ndarray, targets:np.ndarray) -> Iterator:
        starts=np.arange(0, len(inputs), self.batch_size)
        if self.shuffle==True:
            np.random.shuffle(starts)
        
        for start in starts:
            end=start+self.batch_size
            batch_inputs=inputs[start:end]
            batch_targets=targets[start:end]
            yield Batch(batch_inputs, batch_targets)