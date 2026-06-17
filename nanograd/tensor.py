import numpy as np
import math
import random
import matplotlib.pyplot as plt 

class Tensor:
    def __init__(self, data: np.ndarray, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = np.zeros(self.data.shape)
        self.label = label
        self._backward = lambda: None
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), '+', label=f'({self.label}+{other.label})')
        def _backward():
            self.grad += 1.0 *  out.grad
            other.grad += 1.0 *  out.grad
        out._backward = _backward
                
        return out
        
    
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*', label=f'({self.label}*{other.label})')
        def _backward():
            self.grad +=  other.data * out.grad
            other.grad +=  self.data * out.grad
            
        out._backward = _backward    
        
        return out

    def exp(self):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(math.exp(self.data), (self,), 'exp', label=f'exp({self.label})')
        
        def _backward():
            self.grad +=  out.data * out.grad
            
        out._backward = _backward    
        
        return out
        
    
    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum', label=f'sum({self.label})')
        
        def _backward():
            self.grad += 1.0 *  out.grad
        out._backward = _backward
        
        return out

    
    def topo_sort(self):
        visited = set()
        topo = []      
          
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        return topo
    
            
    
    def backward(self):
        if len(self._prev) == 0:
            self.grad = np.ones(self.data.shape) 
            return
        
        topo = self.topo_sort()
        
        self.grad = np.ones(self.data.shape)
        for i in reversed(topo):
            i._backward()
    
        # bfs = [self]
        
        # while len(bfs) > 0:
        #     i = bfs[0]
        #     if len(i._prev) == 0:
        #         bfs.remove(i)
        #         continue
            
        #     h = 0.00001
        #     x1, x2 = i._prev
        #     if i._op == '+':
        #         x = x1.data + x2.data + h
        #         x1.grad = ((x - i.data) / h) * i.grad
        #         x2.grad = ((x - i.data) / h) * i.grad

        #     elif i._op == '*':
        #         derev_help_1 = (x1.data + h) * x2.data 
        #         derev_help_2 = x1.data * (x2.data + h)
                
        #         x1.grad = ((derev_help_1 - i.data) / h) * i.grad
        #         x2.grad = ((derev_help_2 - i.data) / h) * i.grad
                
        #     bfs.append(x1)
        #     bfs.append(x2)
        #     bfs.remove(i)
            
    # def step(self, step_size=0.01):
    #     bfs = [list(self._prev)]
    #     while len(bfs) > 0:
    #         i = bfs [0]
    #         i.data = step_size * i.grad
    #         bfs.remove(i)
    #         bfs.append(list(i._prev))
                      

if __name__ == "__main__":

    # napisz kod który sprawdza czy dobrze licze gradienty i łatwo się sprawdza
    a = Tensor(np.array([[1, 2], [3, 4]]), label='a')
    b = Tensor(np.array([[5, 6], [7, 8]]), label='b')
    c = Tensor(np.array([[1, 2], [1, 1]]), label='c')

    # d  = a + b

    e = c * (a + b)

    lists = [a, b, c, e]
    lists[-1].backward()
    for i in lists:
        print("\n")
        print(i.grad)

    print("\n")
