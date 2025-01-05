import math

from modularized_implementation.topological_sort import topological_sort
from typing import List, Tuple, Union

class Value:
    def __init__(self, data: float, label: str="", operation: str=None, children: Tuple=None):
        self.data = data
        self.label = label
        self.operation = operation
        self.children = children
        self.grad = 0.0
        self.grad_propagator = None
    
    def __add__(self, other: Union['Value', float, int]) -> 'Value':
        if isinstance(other, float) or isinstance(other, int):
            other = Value(other)
        result = Value(self.data + other.data, operation="+", children=(self, other))
        def grad_propagator():
            self.grad += result.grad
            other.grad += result.grad
        result.grad_propagator = grad_propagator
        return result
    
    def __radd__(self, other: Union[float, int]) -> 'Value':
        other: Value = Value(other)
        result = Value(self.data + other.data, operation="+", children=(self, other))
        def grad_propagator():
            self.grad += result.grad
            other.grad += result.grad
        result.grad_propagator = grad_propagator
        return result
    
    def __mul__(self, other: Union['Value', float, int]) -> 'Value':
        if isinstance(other, float) or isinstance(other, int):
            other = Value(other)
        result = Value(self.data * other.data, operation="*", children=(self, other)) 
        def grad_propagator():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
        result.grad_propagator = grad_propagator
        return result
    
    def __rmul__(self, other: Union[float, int]) -> 'Value':
        other = Value(other)
        result = Value(self.data * other.data, operation="*", children=(self, other)) 
        def grad_propagator():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
        result.grad_propagator = grad_propagator
        return result

    def __sub__(self, other: Union['Value', float, int]) -> 'Value':
        if isinstance(other, float) or isinstance(other, int):
            other = Value(other)
        result = Value(self.data - other.data, operation=f"{self.data}-{other.data}", children=(self, other))
        def grad_propagator():
            self.grad += result.grad
            other.grad -= result.grad
        result.grad_propagator = grad_propagator
        return result

    def __rsub__(self, other: Union[float, int]) -> 'Value':
        other = Value(other)
        result = Value(other.data - self.data, operation=f"{other.data}-{self.data}", children=(other, self))
        def grad_propagator():
            self.grad -= result.grad
            other.grad += result.grad
        result.grad_propagator = grad_propagator
        return result

    def __truediv__(self, other: Union['Value', float, int]) -> 'Value':
        if isinstance(other, float) or isinstance(other, int):
            other = Value(other)
        result = Value(self.data / other.data, operation=f"{self.data}/{other.data}", children=(self, other))
        def grad_propagator():
            self.grad += result.grad / other.data
            other.grad -= ((result.grad * self.data) / other.data**2)
        result.grad_propagator = grad_propagator
        return result

    def __pow__(self, other: Union['Value', float]) -> 'Value':
        if isinstance(other, float) or isinstance(other, int):
            other = Value(other)
        result = Value(self.data ** other.data, operation=f"{self.data}^{other.data}", children=(self, other))
        def grad_propagator():
            self.grad += result.grad * other.data * self.data**(other.data - 1)
            other.grad += result.grad * result.data * math.log(self.data)
        result.grad_propagator = grad_propagator
        return result

    def __rpow__(self, other: Union[int, float]) -> 'Value':
        other = Value(other)
        result = Value(other.data ** self.data, operation=f"{other.data}^{self.data}", children=(other, self))
        def grad_propagator():
            self.grad += result.grad * result.data * math.log(other.data)
            other.grad += result.grad * self.data * other.data**(self.data - 1)
        result.grad_propagator = grad_propagator
        return result

    def tanh(self) -> 'Value':
        computed_data = (math.exp(self.data) - math.exp(-self.data))/(math.exp(self.data) + math.exp(-self.data))
        result = Value(data=computed_data, operation="tanh", children=(self,))
        def grad_propagator():
            # The derivative of tanh is (1 - tanh^2). This is used to calculate the gradients of the children nodes.
            self.grad += (1.0 - computed_data**2) * result.grad
        result.grad_propagator = grad_propagator
        return result

    def backward(self):
        topo_order: List[Value] = topological_sort(self)
        self.grad = 1.0
        for node in topo_order:
            if node.grad_propagator is None:
                continue
            node.grad_propagator()

    def __repr__(self) -> str:
        return f"label: {self.label} | data: {self.data} | operation: {self.operation} | grad: {self.grad}"