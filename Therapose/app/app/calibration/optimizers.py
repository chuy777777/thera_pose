import numpy as np

class Adagrad():
    def __init__(self, params_length, alpha, eps=1e-8):
        self.params_length=params_length
        self.alpha=alpha
        self.eps=eps
        
        self.init()

    def init(self):
        self.accum=np.zeros(self.params_length)

    def iterate(self, params, grad):
        self.accum=self.accum + grad ** 2
        beta=self.alpha / (np.sqrt(self.accum) + self.eps)
        new_params=params - beta * grad
        return new_params
    
class RMSprop():
    # momentum: (0.9, 0.99, 0.5)
    def __init__(self, params_length, alpha, momentum=0.9, eps=1e-8):
        self.params_length=params_length
        self.alpha=alpha
        self.momentum=momentum # beta
        self.eps=eps
        
        self.init()

    def init(self):
        self.predicted_square_grad=np.zeros(self.params_length)

    def iterate(self, params, grad):
        self.predicted_square_grad=self.momentum * self.predicted_square_grad + (1 - self.momentum) * (grad ** 2)
        new_params=params - (self.alpha / (np.sqrt(self.predicted_square_grad) + self.eps)) * grad
        return new_params
    
class AdaDelta():
    # momentum: (0.9, 0.99, 0.5)
    def __init__(self, params_length, init_val_delta_param=0.1, momentum=0.9, eps=1e-8):
        self.params_length=params_length
        self.init_val_delta_param=init_val_delta_param
        self.momentum=momentum # beta
        self.eps=eps
        
        self.init()

    def init(self):
        self.delta_param=np.ones(self.params_length) * self.init_val_delta_param
        self.predicted_square_delta_param=np.zeros(self.params_length)
        self.predicted_square_grad=np.zeros(self.params_length)

    def iterate(self, params, grad):
        self.predicted_square_delta_param=self.momentum * self.predicted_square_delta_param + (1 - self.momentum) * (self.delta_param ** 2)
        self.predicted_square_grad=self.momentum * self.predicted_square_grad + (1 - self.momentum) * (grad ** 2)
        self.delta_param=-(np.sqrt(self.predicted_square_delta_param + self.eps) / np.sqrt(self.predicted_square_grad + self.eps)) * grad
        new_params=params + self.delta_param
        return new_params
    
class Adam():
    def __init__(self, params_length, alpha, beta_1=0.9, beta_2=0.99, eps=1e-8):
        self.params_length=params_length
        self.alpha=alpha
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.eps=eps
        
        self.init()
        
    def init(self):
        self.it=0
        self.predicted_grad=np.zeros(self.params_length)
        self.predicted_square_grad=np.zeros(self.params_length)
        
    def iterate(self, params, grad):
        self.it+=1
        self.predicted_grad=self.beta_1 * self.predicted_grad + (1 - self.beta_1) * grad
        self.predicted_square_grad=self.beta_2 * self.predicted_square_grad + (1 - self.beta_2) * (grad ** 2)
        best_predicted_grad=self.predicted_grad / (1 - self.beta_1 ** self.it)
        best_predicted_square_grad=self.predicted_square_grad / (1 - self.beta_2 ** self.it)
        new_params=params - (self.alpha / np.sqrt(best_predicted_square_grad + self.eps)) * best_predicted_grad
        return new_params
