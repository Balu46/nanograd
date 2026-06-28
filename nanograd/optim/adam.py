from nanograd.core.backend import get_xp

class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.
    """
    def __init__(self, model_params: list, learning_rate: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.t = 0
        self.model_params = model_params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = []
        self.v = []
        for param in self.model_params:
            xp = get_xp(param.data)
            self.m.append(xp.zeros(param.data.shape))
            self.v.append(xp.zeros(param.data.shape))
        
    def step(self):
        self.t += 1
        for i in range(len(self.model_params)):
            xp = get_xp(self.model_params[i].data)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.model_params[i].grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (self.model_params[i].grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            self.model_params[i].data = self.model_params[i].data - self.learning_rate * m_hat / (xp.sqrt(v_hat) + self.epsilon)
            
    def zero_grad(self):
        for param in self.model_params:
            xp = get_xp(param.data)
            param.grad = xp.zeros(param.data.shape)
