import torch
from nanograd.tensor import Tensor

def test_tensor_addition():
    # Twój silnik
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a + b
    
    # PyTorch jako ostateczna wyrocznia
    a_t = torch.tensor([1.0, 2.0], requires_grad=True)
    b_t = torch.tensor([3.0, 4.0], requires_grad=True)
    c_t = a_t + b_t
    
    # Test forward passu
    assert np.allclose(c.data, c_t.detach().numpy())
    
    # Test backward passu (sztuczny gradient zewnętrzny)
    c_t.backward(torch.tensor([1.0, 1.0]))
    
    # Tę metodę musisz wymyślić i zaimplementować sam
    c.backward(np.array([1.0, 1.0]))
    
    assert np.allclose(a.grad, a_t.grad.numpy())
    assert np.allclose(b.grad, b_t.grad.numpy())