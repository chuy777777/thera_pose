import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

"""
COMPONENTES GRAFO COMPUTACIONAL
"""

class Node():
    def __init__(self, derivative_list, child_list, calculate_grad=False):
        self.derivative_list=derivative_list
        self.child_list=child_list
        self.calculate_grad=calculate_grad
        self.grad=0 

    def __str__(self):
        return "derivative_list: {}\nchild_list: {}\ncalculate_grad: {}\ngrad: {}\n\n".format(self.derivative_list, self.child_list, self.calculate_grad, self.grad)

class Chuy():
    def __init__(self, array, calculate_grad, node_array=None):
        self.array: np.ndarray=array
        self.calculate_grad=calculate_grad

        if node_array is None:
            m,n=array.shape
            self.node_array=np.zeros((m,n), dtype=object)
            for i in range(m):
                for j in range(n):
                    self.node_array[i,j]=Node(derivative_list=[], child_list=[], calculate_grad=calculate_grad)
        else:
            self.node_array=node_array
    
    def iterative_backward(self, node: Node):
        stack_childs=[]
        stack_accum=[]
        for i in range(len(node.child_list)):
            stack_childs.append(node.child_list[i])
            stack_accum.append(node.derivative_list[i])
        while len(stack_childs) > 0:
            n=stack_childs.pop()
            accum=stack_accum.pop()
            if len(n.child_list) > 0:
                for i in range(len(n.child_list)):
                    stack_childs.append(n.child_list[i])
                    stack_accum.append(accum * n.derivative_list[i])
            else:
                if n.calculate_grad:
                    n.grad+=accum
    
    def recursive_backward(self, node: Node, accum=1):
        if len(node.child_list) > 0:
            for i in range(len(node.child_list)):
                self.recursive_backward(node=node.child_list[i], accum=accum * node.derivative_list[i])
        else:
            if node.calculate_grad:
                node.grad+=accum
    
    def backward(self):
        m,n=self.node_array.shape
        for i in range(m):
            for j in range(n):
                self.iterative_backward(node=self.node_array[i,j])
                # self.recursive_backward(node=self.node_array[i,j])
    
    def transpose(self):
        return Chuy(array=self.array.T, calculate_grad=self.calculate_grad, node_array=self.node_array.T)
    
    def get_grad(self):
        m,n=self.node_array.shape
        grad=np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                grad[i,j]=self.node_array[i,j].grad
        return grad
    
    # + Addition (a + b)
    def __add__(self, other):
        m,n=self.array.shape
        calculate_grad=False
        array=None
        node_array=np.zeros(self.array.shape, dtype=object)
        if isinstance(other, int) or isinstance(other, float):
            array=self.array + other
            for i in range(m):
                for j in range(n):
                    node_array[i,j]=Node(derivative_list=[1], child_list=[self.node_array[i,j]], calculate_grad=calculate_grad)
        else:
            array=self.array + other.array
            for i in range(m):
                for j in range(n):
                    node_array[i,j]=Node(derivative_list=[1,1], child_list=[self.node_array[i,j],other.node_array[i,j]], calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)
    
    # - Subtraction (a - b)
    def __sub__(self, other):
        m,n=self.array.shape
        calculate_grad=False
        array=None
        node_array=np.zeros(self.array.shape, dtype=object)
        if isinstance(other, int) or isinstance(other, float):
            array=self.array - other
            for i in range(m):
                for j in range(n):
                    node_array[i,j]=Node(derivative_list=[1], child_list=[self.node_array[i,j]], calculate_grad=calculate_grad)
        else:
            array=self.array - other.array
            for i in range(m):
                for j in range(n):
                    node_array[i,j]=Node(derivative_list=[1,-1], child_list=[self.node_array[i,j],other.node_array[i,j]], calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)
    
    # * Multiplication (a * b)
    def __mul__(self, other):
        m,n=self.array.shape
        calculate_grad=False
        array=None
        node_array=None
        if isinstance(other, int) or isinstance(other, float):
            array=self.array * other
            node_array=np.zeros(array.shape, dtype=object)
            for i in range(m):
                for j in range(n):
                    node_array[i,j]=Node(derivative_list=[other], child_list=[self.node_array[i,j]], calculate_grad=calculate_grad)
        elif isinstance(other, np.ndarray) and other.dtype != object:
            p,q=other.shape
            hs,ws=m*p,n*q
            array=np.zeros((hs,ws))
            node_array=np.zeros(array.shape, dtype=object)
            for i in range(hs):
                for j in range(ws):
                    array[i,j]=self.array[i%m,j%n] * other[i//m,j//n]
                    node_array[i,j]=Node(derivative_list=[other[i//m,j//n]], child_list=[self.node_array[i%m,j%n]], calculate_grad=calculate_grad)                    
        else:
            array=self.array * other.array
            node_array=np.zeros(array.shape, dtype=object)
            for i in range(m):
                for j in range(n):
                    node_array[i,j]=Node(derivative_list=[other.array[i,j],self.array[i,j]], child_list=[self.node_array[i,j],other.node_array[i,j]], calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)

    # @ Matrix multiplication (a @ b)
    def __matmul__(self, other):
        calculate_grad=False
        array=self.array @ other.array
        node_array=np.zeros(array.shape, dtype=object)
        m,n=array.shape
        for i in range(m):
            for j in range(n):
                node_array[i,j]=Node(derivative_list=other.array[:,j].tolist() + self.array[i,:].tolist(), child_list=self.node_array[i,:].tolist() + other.node_array[:,j].tolist(), calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)

    # / Division (a / b)
    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__mul__(other=1 / other)
        else:
            m,n=self.array.shape
            p,q=other.array.shape
            calculate_grad=False
            array=self.array / other.array
            node_array=np.zeros(array.shape, dtype=object)
            if m == p and n == q:
                # self.array.shape = other.array.shape
                for i in range(m):
                    for j in range(n):
                        node_array[i,j]=Node(derivative_list=[1 / other.array[i,j],-(self.array[i,j] / (other.array[i,j] ** 2))], child_list=[self.node_array[i,j],other.node_array[i,j]], calculate_grad=calculate_grad)
            elif p == m and q == 1:
                # other.array es un vector columna
                for i in range(m):
                    for j in range(n):
                        node_array[i,j]=Node(derivative_list=[1 / other.array[i,0],-(self.array[i,j] / (other.array[i,0] ** 2))], child_list=[self.node_array[i,j],other.node_array[i,0]], calculate_grad=calculate_grad)
            elif p == 1 and q == n:
                # other.array es un vector fila
                for i in range(m):
                    for j in range(n):
                        node_array[i,j]=Node(derivative_list=[1 / other.array[0,j],-(self.array[i,j] / (other.array[0,j] ** 2))], child_list=[self.node_array[i,j],other.node_array[0,j]], calculate_grad=calculate_grad)
            return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)
            
    # ** Power (a ** b)
    def __pow__(self, module):
        m,n=self.array.shape
        calculate_grad=False
        array=self.array ** float(module)
        node_array=np.zeros(array.shape, dtype=object)
        for i in range(m):
            for j in range(n):
                node_array[i,j]=Node(derivative_list=[module * (self.array[i,j] ** float(module - 1))], child_list=[self.node_array[i,j]], calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)

    # Siempre devolvera un vector/matriz con dimensions (p,q)
    def __getitem__(self, key):
        temp_array=self.array[key]
        temp_node_array=self.node_array[key]
        if not isinstance(temp_array, np.ndarray):
            temp_array=np.array([[temp_array]])
            temp_node_array=np.array([[temp_node_array]])
        elif len(temp_array.shape) == 1:
            if isinstance(key[0], int) or (isinstance(key[0], list) and len(key[0]) == 1):
                temp_array=temp_array[None,:]
                temp_node_array=temp_node_array[None,:]
            elif isinstance(key[1], int) or (isinstance(key[1], list) and len(key[1]) == 1): 
                temp_array=temp_array[:,None]
                temp_node_array=temp_node_array[:,None]
        return Chuy(array=temp_array, calculate_grad=self.calculate_grad, node_array=temp_node_array)
    
    def __setitem__(self, key):
        pass
    
    # Otros metodos
    def exp(self):
        array=np.exp(self.array)
        m,n=array.shape
        calculate_grad=False
        node_array=np.zeros(array.shape, dtype=object)
        for i in range(m):
            for j in range(n):
                node_array[i,j]=Node(derivative_list=[np.exp(self.array[i,j])], child_list=[self.node_array[i,j]], calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)

    def log(self):
        array=np.log(self.array)
        m,n=array.shape
        calculate_grad=False
        node_array=np.zeros(array.shape, dtype=object)
        for i in range(m):
            for j in range(n):
                node_array[i,j]=Node(derivative_list=[1 / self.array[i,j]], child_list=[self.node_array[i,j]], calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)
    
    def sin(self):
        array=np.sin(self.array)
        m,n=array.shape
        calculate_grad=False
        node_array=np.zeros(array.shape, dtype=object)
        for i in range(m):
            for j in range(n):
                node_array[i,j]=Node(derivative_list=[np.cos(self.array[i,j])], child_list=[self.node_array[i,j]], calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)

    # Devuelve un objeto 'Chuy' con array de tamano (1,1), (1,n), (m,1), respectivamente
    def sum(self, axis=None, out=None, **kwargs):
        # NumPy array
        # axis=None -> En todo
        # axis=0 -> A lo largo de las columnas
        # axis=1 -> A lo largo de las filas
        m,n=self.array.shape
        calculate_grad=False
        array=None
        node_array=None
        if axis is None:
            array=np.array([[self.array.sum()]])
            node_array=np.zeros(array.shape, dtype=object)
            node_array[0,0]=Node(derivative_list=np.ones((m * n)).tolist(), child_list=self.node_array.flatten(order='C').tolist(), calculate_grad=calculate_grad)
        elif axis == 0:
            array=np.zeros((1,n))
            array[0,:]=self.array.sum(axis=0)
            node_array=np.zeros(array.shape, dtype=object)
            for j in range(n):
                node_array[0,j]=Node(derivative_list=np.ones((m)).tolist(), child_list=self.node_array[:,j].tolist(), calculate_grad=calculate_grad)
        elif axis == 1:
            array=np.zeros((m,1))
            array[:,0]=self.array.sum(axis=1)
            node_array=np.zeros(array.shape, dtype=object)
            for i in range(m):
                node_array[i,0]=Node(derivative_list=np.ones((n)).tolist(), child_list=self.node_array[i,:].tolist(), calculate_grad=calculate_grad)
        return Chuy(array=array, calculate_grad=calculate_grad, node_array=node_array)

"""
NORMALIZACION DE DATOS

PARA:
    - REGRESION
        SE HACE PARA TODOS LOS DATOS
    - CLASIFICACION
        SE HACE PARA CADA CLASE (NO)
        
        SE HACE PARA TODOS LOS DATOS
            YA QUE SI SE TIENE UN EJEMPLO NUEVO NO SE SABE LA CLASE A LA QUE PERTENECE
"""

class StandardNormalization():
    @staticmethod 
    def transform(data, mean_std=None):
        m,n=data.shape 
        data_mean,data_std=None,None
        if mean_std is None:
            data_mean,data_std=data.mean(axis=0),data.std(axis=0, ddof=0) 
        else:
            data_mean,data_std=mean_std[0],mean_std[1]
        data_norm=np.zeros((m,n))
        for col in range(n):
            if data_std[col] != 0:
                data_norm[:,col]=(data[:,col] - data_mean[col]) / data_std[col]
            else:
                data_norm[:,col]=data[:,col]
        return data_norm,data_mean,data_std

"""
FUNCIONES DE PERDIDA
"""

class MSELossFunction():
    def __init__(self, layer_list):
        self.layer_list=layer_list
        self.c_cost=None
    
    def get_grad_shape(self):
        p=0
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            p+=layer.params_length
        return p

    def get_grad(self):
        grad=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            grad=layer.get_grad() if grad is None else np.concatenate([grad, layer.get_grad()], axis=0)
        return grad
    
    def get_param_vals(self):
        # Obtenemos el valor de los parametros
        param_vals=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            param_vals=layer.get_param_vals() if param_vals is None else np.concatenate([param_vals, layer.get_param_vals()], axis=0)
        return param_vals
    
    def set_param_vals(self, param_vals):
        # Cambiamos el valor de los parametros
        p=0
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            p+=layer.params_length
            layer.set_param_vals(param_vals=param_vals[p - layer.params_length:p,[0]])
                    
    def forward(self, c_X_batch, Y_batch):
        c_output_data=self.predict(c_X_batch=c_X_batch)
        m,_=c_output_data.array.shape
        c_Y_batch=Chuy(array=Y_batch, calculate_grad=False)
        c_L=(c_output_data - c_Y_batch).transpose()@(c_output_data - c_Y_batch)
        self.c_cost=c_L *  ((1 / 2) * m)
        return c_output_data
    
    def backward(self):
        self.c_cost.backward()
        
    def predict(self, c_X_batch):
        c_output_data=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            if i == 0:
                c_output_data=layer.set_input_data_and_get_output_data(c_input_data=c_X_batch)
            else:
                c_output_data=layer.set_input_data_and_get_output_data(c_input_data=c_output_data)
        return c_output_data
    
    def get_predicted_output(self, c_output_data):
        Y_predicted=c_output_data.array
        return Y_predicted
    
class BCELossFunction():
    def __init__(self, layer_list):
        self.layer_list=layer_list
        self.c_cost=None
    
    def get_grad_shape(self):
        p=0
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            p+=layer.params_length
        return p

    def get_grad(self):
        grad=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            grad=layer.get_grad() if grad is None else np.concatenate([grad, layer.get_grad()], axis=0)
        return grad
    
    def get_param_vals(self):
        # Obtenemos el valor de los parametros
        param_vals=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            param_vals=layer.get_param_vals() if param_vals is None else np.concatenate([param_vals, layer.get_param_vals()], axis=0)
        return param_vals
    
    def set_param_vals(self, param_vals):
        # Cambiamos el valor de los parametros
        p=0
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            p+=layer.params_length
            layer.set_param_vals(param_vals=param_vals[p - layer.params_length:p,[0]])
                    
    def forward(self, c_X_batch, Y_batch):
        c_output_data=self.predict(c_X_batch=c_X_batch)
        m,_=c_output_data.array.shape
        c_Y_batch=Chuy(array=Y_batch, calculate_grad=False)
        c_C=np.log(c_output_data)
        c_D=np.log((c_output_data * -1) + 1)
        c_E=(c_Y_batch * -1) + 1
        c_L=((c_Y_batch.transpose() @ c_C) * -1) + ((c_E.transpose() @ c_D) * -1)
        self.c_cost=c_L * (1 / m)
        return c_output_data
    
    def backward(self):
        self.c_cost.backward()
        
    def predict(self, c_X_batch):
        c_output_data=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            if i == 0:
                c_output_data=layer.set_input_data_and_get_output_data(c_input_data=c_X_batch)
            else:
                c_output_data=layer.set_input_data_and_get_output_data(c_input_data=c_output_data)
        return c_output_data

    def get_predicted_output(self, c_output_data):
        output_data=c_output_data.array        
        # Nos quedamos con la clase que sea mayor o igual al umbral establecido 
        umbral=0.5
        Y_predicted=(output_data >= umbral).astype(int)
        return Y_predicted
    
class CELossFunction():
    def __init__(self, layer_list):
        self.layer_list=layer_list
        self.c_cost=None
    
    def get_grad_shape(self):
        p=0
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            p+=layer.params_length
        return p

    def get_grad(self):
        grad=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            grad=layer.get_grad() if grad is None else np.concatenate([grad, layer.get_grad()], axis=0)
        return grad
    
    def get_param_vals(self):
        # Obtenemos el valor de los parametros
        param_vals=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            param_vals=layer.get_param_vals() if param_vals is None else np.concatenate([param_vals, layer.get_param_vals()], axis=0)
        return param_vals
    
    def set_param_vals(self, param_vals):
        # Cambiamos el valor de los parametros
        p=0
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            p+=layer.params_length
            layer.set_param_vals(param_vals=param_vals[p - layer.params_length:p,[0]])
                    
    def forward(self, c_X_batch, Y_batch):
        c_output_data=self.predict(c_X_batch=c_X_batch)
        m,_=c_output_data.array.shape
        c_L=None
        for i in range(m):
            c_L=np.log(c_output_data[i,Y_batch[i,0]]) if c_L is None else c_L + np.log(c_output_data[i,Y_batch[i,0]])
        self.c_cost=c_L * -(1 / m)
        return c_output_data
    
    def backward(self):
        self.c_cost.backward()
        
    def predict(self, c_X_batch):
        c_output_data=None
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            if i == 0:
                c_output_data=layer.set_input_data_and_get_output_data(c_input_data=c_X_batch)
            else:
                c_output_data=layer.set_input_data_and_get_output_data(c_input_data=c_output_data)
        return c_output_data
    
    def get_predicted_output(self, c_output_data):
        output_data=c_output_data.array        
        # Nos quedamos con la clase con la mayor probabilidad 
        # Devuelve la posicion del primero que encuentra (axis: 0 (columnas) 1 (filas))
        Y_predicted=np.unravel_index(np.argmax(output_data, axis=1), output_data.shape)[1][:,None]
        return Y_predicted

"""
MODIFICADORES DE TASA DE APRENDIZAJE
"""

# Lo que hace es modificar la tasa de aprendizaje (alpha) del optimizador en cada iteracion
# Para n muy grande se comporta como la funcion de una recta
class ExponentialDecayLRSchedule():
    def __init__(self, optimizer, epochs, min_alpha=1e-6, n=1):
        self.optimizer=optimizer
        self.epochs=epochs
        self.min_alpha=min_alpha
        self.n=n
        
        # Construimos la funcion
        alpha_ini,alpha_fin=self.optimizer.alpha,self.min_alpha
        self.beta=(alpha_fin / alpha_ini) ** (1 / self.n)
        self.c=np.log(alpha_ini) / np.log(self.beta)
        self.delta_x=self.n / (self.epochs - 1)
        
    def f(self, x):
        return self.beta ** (x + self.c)
        
    def change(self, iteration):
        alpha_i=self.f(x=0 + self.delta_x * iteration)
        self.optimizer.alpha=alpha_i

"""
OPTIMIZADORES
"""

# Trabaja con 'minibatches'
class MBGDOptimizer():
    # momentum: (0.9, 0.99, 0.5)
    def __init__(self, loss, alpha, momentum=0.9, nesterov=False):
        self.loss=loss
        self.alpha=alpha
        self.momentum=momentum # beta
        self.nesterov=nesterov
        
        self.init()

    def init(self):
        self.it=0
        self.u=np.zeros(self.loss.get_grad_shape())
        self.predicted_grad=np.zeros(self.loss.get_grad_shape())

    def iterate(self, c_X_batch, Y_batch):
        self.it+=1
        if self.nesterov:
            if self.it == 1:
                self.loss.forward(c_X_batch=c_X_batch, Y_batch=Y_batch)
                self.loss.backward()
                g=self.loss.get_grad()
                self.loss.set_param_vals(param_vals=self.loss.get_param_vals() + -(self.momentum * g))
                self.loss.forward(c_X_batch=c_X_batch, Y_batch=Y_batch)
                self.loss.backward()
                g_next=self.loss.get_grad()
                self.u=-(self.momentum * g) + -(self.alpha * g_next)
                param_vals=self.loss.get_param_vals() + self.u
                self.loss.set_param_vals(param_vals=param_vals)
            else:
                self.loss.set_param_vals(param_vals=self.loss.get_param_vals() + self.u)
                self.loss.forward(c_X_batch=c_X_batch, Y_batch=Y_batch)
                self.loss.backward()
                self.u=self.u - self.alpha * self.loss.get_grad()
                param_vals=self.loss.get_param_vals() + self.u
                self.loss.set_param_vals(param_vals=param_vals)
        else:        
            self.loss.forward(c_X_batch=c_X_batch, Y_batch=Y_batch)
            self.loss.backward()
            self.predicted_grad=self.momentum * self.predicted_grad + (1 - self.momentum) * self.loss.get_grad()
            param_vals=self.loss.get_param_vals() - self.alpha * self.predicted_grad
            self.loss.set_param_vals(param_vals=param_vals)  
        return self.loss.c_cost.array[0,0]
    
    def get_dict_configurations(self):
        dict_configurations={
            "alpha": self.alpha,
            "momentum": self.momentum,
            "nesterov": self.nesterov
        }
        return dict_configurations

class AdagradOptimizer():
    def __init__(self, loss, alpha, eps=1e-8):
        self.loss=loss
        self.alpha=alpha
        self.eps=eps
        
        self.init()

    def init(self):
        self.accum=np.zeros(self.loss.get_grad_shape())

    def iterate(self, c_X_batch, Y_batch):
        self.loss.forward(c_X_batch=c_X_batch, Y_batch=Y_batch)
        self.loss.backward()
        self.accum=self.accum + self.loss.get_grad() ** 2
        beta=self.alpha / (np.sqrt(self.accum) + self.eps)
        param_vals=self.loss.get_param_vals() - beta * self.loss.get_grad()
        self.loss.set_param_vals(param_vals=param_vals) 
        return self.loss.c_cost.array[0,0]
    
    def get_dict_configurations(self):
        dict_configurations={
            "alpha": self.alpha,
            "eps": self.eps
        }
        return dict_configurations
    
class RMSpropOptimizer():
    # momentum: (0.9, 0.99, 0.5)
    def __init__(self, loss, alpha, momentum=0.9, eps=1e-8):
        self.loss=loss
        self.alpha=alpha
        self.momentum=momentum # beta
        self.eps=eps
        
        self.init()

    def init(self):
        self.predicted_square_grad=np.zeros(self.loss.get_grad_shape())

    def iterate(self, c_X_batch, Y_batch):
        self.loss.forward(c_X_batch=c_X_batch, Y_batch=Y_batch)
        self.loss.backward()
        self.predicted_square_grad=self.momentum * self.predicted_square_grad + (1 - self.momentum) * (self.loss.get_grad() ** 2)
        param_vals=self.loss.get_param_vals() - (self.alpha / (np.sqrt(self.predicted_square_grad) + self.eps)) * self.loss.get_grad()
        self.loss.set_param_vals(param_vals=param_vals) 
        return self.loss.c_cost.array[0,0] 
    
    def get_dict_configurations(self):
        dict_configurations={
            "alpha": self.alpha,
            "momentum": self.momentum,
            "eps": self.eps
        }
        return dict_configurations

class AdaDeltaOptimizer():
    # momentum: (0.9, 0.99, 0.5)
    def __init__(self, loss, init_val_delta_param=0.1, momentum=0.9, eps=1e-8):
        self.loss=loss
        self.init_val_delta_param=init_val_delta_param
        self.momentum=momentum # beta
        self.eps=eps
        
        self.init()

    def init(self):
        self.delta_param=np.ones(self.loss.get_grad_shape()) * self.init_val_delta_param
        self.predicted_square_delta_param=np.zeros(self.loss.get_grad_shape())
        self.predicted_square_grad=np.zeros(self.loss.get_grad_shape())

    def iterate(self, c_X_batch, Y_batch):
        self.loss.forward(c_X_batch=c_X_batch, Y_batch=Y_batch)
        self.loss.backward()
        self.predicted_square_delta_param=self.momentum * self.predicted_square_delta_param + (1 - self.momentum) * (self.delta_param ** 2)
        self.predicted_square_grad=self.momentum * self.predicted_square_grad + (1 - self.momentum) * (self.loss.get_grad() ** 2)
        self.delta_param=-(np.sqrt(self.predicted_square_delta_param + self.eps) / np.sqrt(self.predicted_square_grad + self.eps)) * self.loss.get_grad()
        param_vals=self.loss.get_param_vals() + self.delta_param
        self.loss.set_param_vals(param_vals=param_vals)  
        return self.loss.c_cost.array[0,0] 
    
    def get_dict_configurations(self):
        dict_configurations={
            "init_val_delta_param": self.init_val_delta_param,
            "momentum": self.momentum,
            "eps": self.eps
        }
        return dict_configurations
           
class AdamOptimizer():
    def __init__(self, loss, alpha, beta_1=0.9, beta_2=0.99, eps=1e-8):
        self.loss=loss
        self.alpha=alpha
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.eps=eps
        
        self.init()
        
    def init(self):
        self.it=0
        self.predicted_grad=np.zeros(self.loss.get_grad_shape())
        self.predicted_square_grad=np.zeros(self.loss.get_grad_shape())
        
    def iterate(self, c_X_batch, Y_batch):
        self.it+=1
        self.loss.forward(c_X_batch=c_X_batch, Y_batch=Y_batch)
        self.loss.backward()
        self.predicted_grad=self.beta_1 * self.predicted_grad + (1 - self.beta_1) * self.loss.get_grad()
        self.predicted_square_grad=self.beta_2 * self.predicted_square_grad + (1 - self.beta_2) * (self.loss.get_grad() ** 2)
        best_predicted_grad=self.predicted_grad / (1 - self.beta_1 ** self.it)
        best_predicted_square_grad=self.predicted_square_grad / (1 - self.beta_2 ** self.it)
        param_vals=self.loss.get_param_vals() - (self.alpha / np.sqrt(best_predicted_square_grad + self.eps)) * best_predicted_grad
        self.loss.set_param_vals(param_vals=param_vals)  
        return self.loss.c_cost.array[0,0]
    
    def get_dict_configurations(self):
        dict_configurations={
            "alpha": self.alpha,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "eps": self.eps
        }
        return dict_configurations

"""
COMPONENTES REDES NEURONALES
"""

# METRICAS DE EVALUACION
# Se espera que las clases esten del 0 a c-1
class F1ScoreEvaluationMetric():
    def __init__(self, number_classes):
        self.number_classes=number_classes
        self.confusion_matrix=None
        self.f1_score=None
    
    # (m,1), (m,1)
    def operate(self, Y_predicted, Y_real):
        m,_=Y_predicted.shape
        self.confusion_matrix=np.zeros((self.number_classes,self.number_classes))
        for i in range(m):
            self.confusion_matrix[Y_real[i,0],Y_predicted[i,0]]+=1
        p=np.zeros(self.number_classes)
        r=np.zeros(self.number_classes)
        sum_col=self.confusion_matrix.sum(axis=0)
        sum_row=self.confusion_matrix.sum(axis=1)            
        for i in range(self.number_classes):
            p[i]=0 if sum_col[i] == 0 else self.confusion_matrix[i,i] / sum_col[i]
            r[i]=0 if sum_row[i] == 0 else self.confusion_matrix[i,i] / sum_row[i]
        # f1_score_array=(2 * p * r)/(p + r)
        f1_score_array=np.zeros(self.number_classes)
        for i in range(self.number_classes):
            f1_score_array[i]=0 if p[i] + r[i] == 0 else (2 * p[i] * r[i])/(p[i] + r[i])
        self.f1_score=f1_score_array.mean()

# FUNCIONES DE ACTIVACION
# Se pueden utilizar tanto para objetos de tipo 'Chuy' como de tipo 'Numpy array'
class ActivationFunctions():
    """
    Tenemos:
        - Funcion de actiacion lineal o identidad
        - Funciones de activacion no lineales
    """
    @staticmethod
    def get_activation_function(activation_function_name):
        if activation_function_name == "identity":
            return ActivationFunctions().identity
        elif activation_function_name == "sigmoid":
            return ActivationFunctions().sigmoid
        elif activation_function_name == "softmax":
            return ActivationFunctions().softmax
        else:
            return None
    
    @staticmethod
    def identity(X):
        return X
    
    @staticmethod
    def sigmoid(X):
        # Rango: (0 + eps, 1 - eps)
        eps=1e-8
        c1=1 - 2 * eps
        c2=eps
        S=((np.exp((X * -1)) + 1) ** (-1)) 
        S=(S * c1) + c2
        return S
    
    # @staticmethod
    # def sigmoid(X):
    #     # Rango: (0,1)
    #     return (np.exp((X * -1)) + 1) ** (-1)
    
    @staticmethod
    def softmax(X):
        # Cada valor debe estar en el siguiente rango: (0,1) (debe ser una probabilidad)
        E=np.exp(X)
        Sr=np.sum(E, axis=1, keepdims=True) # A lo largo de las filas 
        S=E / Sr
        return S

# CAPAS DE REDES NEURONALES
class FCLayer():
    def __init__(self, number_neurons, activation_function_name, input_size):
        self.number_neurons=number_neurons
        self.activation_function_name=activation_function_name
        self.input_size=input_size
        self.activation_function=ActivationFunctions.get_activation_function(activation_function_name=self.activation_function_name)
        
        self.c_theta=Chuy(array=np.random.random((self.input_size,self.number_neurons)), calculate_grad=True, node_array=None)
        self.c_bias=Chuy(array=np.random.random((self.number_neurons,1)), calculate_grad=True, node_array=None)
        
        self.params_length=self.input_size * self.number_neurons + self.number_neurons

    def get_param_vals(self):
        param_vals=np.concatenate([self.c_theta.array.flatten(order='C')[:,None], self.c_bias.array], axis=0)
        return param_vals
    
    def set_param_vals(self, param_vals):
        self.c_theta=Chuy(array=np.reshape(param_vals[0:self.input_size * self.number_neurons,[0]], (self.input_size,self.number_neurons), order='C'), calculate_grad=True, node_array=None)
        self.c_bias=Chuy(array=param_vals[self.input_size * self.number_neurons:,[0]], calculate_grad=True, node_array=None)
        
    def get_grad(self):
        # La matriz de parametros se desgloza de izquierda-derecha y arriba-abajo
        grad=np.concatenate([self.c_theta.get_grad().flatten(order='C')[:,None], self.c_bias.get_grad()], axis=0)
        return grad
    
    def set_input_data_and_get_output_data(self, c_input_data):
        m,_=c_input_data.array.shape                   
        c_B=self.c_bias.transpose() * np.ones((m,1))
        c_A=c_input_data @ self.c_theta + c_B
        c_H=self.activation_function(X=c_A)
        return c_H
    
    def get_layer_dict_params(self):
        layer_dict_params={
            "number_neurons": self.number_neurons,
            "activation_function_name": self.activation_function_name,
            "input_size": self.input_size,
            "theta": self.c_theta.array,
            "bias": self.c_bias.array
        }
        return layer_dict_params

    @staticmethod
    def set_layer_dict_params(layer_dict_params):
        layer=FCLayer(number_neurons=layer_dict_params["number_neurons"], activation_function_name=layer_dict_params["activation_function_name"], input_size=layer_dict_params["input_size"])
        layer.c_theta=Chuy(array=layer_dict_params["theta"], calculate_grad=True, node_array=None)
        layer.c_bias=Chuy(array=layer_dict_params["bias"], calculate_grad=True, node_array=None)
        return layer

# CLASE MODELO SECUENCIAL
class SequentialModel():
    def __init__(self):
        self.init()

    def init(self):
        self.layer_list=[]
        self.loss=None
        self.optimizer=None
        self.evaluation_metric=None
        self.loss_name=""
        self.optimizer_name=""
        self.evaluation_metric_name=""

    def add(self, layer):
        self.layer_list.append(layer)
    
    # Todo los nombres en minuscula
    def set_loss(self, loss_name):
        self.loss_name=loss_name
        if loss_name == "mse":
            self.loss=MSELossFunction(layer_list=self.layer_list)
        elif loss_name == "bce":
            self.loss=BCELossFunction(layer_list=self.layer_list)
        elif loss_name == "ce":
            self.loss=CELossFunction(layer_list=self.layer_list)
    
    # Todo los nombres en minuscula
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name=optimizer_name
        if optimizer_name == "mbgd":
            self.optimizer=MBGDOptimizer(loss=self.loss, **kwargs)
        elif optimizer_name == "adagrad":
            self.optimizer=AdagradOptimizer(loss=self.loss, **kwargs)
        elif optimizer_name == "rmsprop":
            self.optimizer=RMSpropOptimizer(loss=self.loss, **kwargs)
        elif optimizer_name == "adadelta":
            self.optimizer=AdaDeltaOptimizer(loss=self.loss, **kwargs)
        elif optimizer_name == "adam":
            self.optimizer=AdamOptimizer(loss=self.loss, **kwargs)
        
    def fit(self, X, Y, batch_size, epochs):
        m,_=X.shape
        number_batches=int(np.ceil(m / batch_size))
        tuple_batches_list=[(Chuy(array=X[i*batch_size:(i+1)*batch_size if i < number_batches - 1 else m,:], calculate_grad=False, node_array=None), Y[i*batch_size:(i+1)*batch_size if i < number_batches - 1 else m,:]) for i in range(number_batches)]
        history=np.zeros((epochs,1))
        self.optimizer.init()
        for i in range(epochs):
            print("Epoch {} of {}".format(i+1,epochs))
            for c_X_batch,Y_batch in tuple_batches_list:
                history[i,0]+=self.optimizer.iterate(c_X_batch=c_X_batch, Y_batch=Y_batch)
        return history
    
    # Todo los nombres en minuscula
    def evaluate(self, X_test, Y_test, evaluation_metric_name="", **kwargs):
        self.evaluation_metric_name=evaluation_metric_name
        # Metricas de evaluacion para clasificacion
        if evaluation_metric_name == "f1_score":
            self.evaluation_metric=F1ScoreEvaluationMetric(**kwargs)

        c_X_test=Chuy(array=X_test, calculate_grad=False)
        c_output_data=self.loss.forward(c_X_batch=c_X_test, Y_batch=Y_test)
        loss_cost=self.loss.c_cost.array[0,0]

        if self.evaluation_metric is not None:
            Y_predicted=self.loss.get_predicted_output(c_output_data=c_output_data)
            self.evaluation_metric.operate(Y_predicted=Y_predicted, Y_real=Y_test)
            return loss_cost,self.evaluation_metric
        
        return loss_cost,None
    
    def save_params(self, path, name, other_configurations={}):
        # Se guarda un diccionario de los parametros entrenados
        full_path="{}/dict_params_model_{}.dat".format(path, name)
        dict_params={"configurations": {
            "loss_name": self.loss_name,
            "optimizer_name": self.optimizer_name,
            "evaluation_metric_name": self.evaluation_metric_name
        }, "layers": {}, "other_configurations": other_configurations}
        for i in range(len(self.layer_list)):
            layer=self.layer_list[i]
            dict_params["layers"]["layer_{}_{}".format(type(layer).__name__,str(i).zfill(2))]={
                "layer_dict_params": layer.get_layer_dict_params()
            }
        if not os.path.exists(path):
            os.makedirs(path)
        with open(full_path, "wb") as f:
            pickle.dump(dict_params, f)

    def load_params(self, path, name):
        self.init()
        full_path="{}/dict_params_model_{}.dat".format(path, name)
        dict_params=None
        try:
            with open(full_path, "rb") as f:
                dict_params=pickle.load(f)
                for key in dict_params["layers"].keys():
                    layer_type=key.split("_")[1]
                    if layer_type == FCLayer.__name__:
                        self.add(layer=FCLayer.set_layer_dict_params(layer_dict_params=dict_params["layers"][key]["layer_dict_params"]))
                self.set_loss(loss_name=dict_params["configurations"]["loss_name"])
                return dict_params
        except (OSError, IOError) as e:
            return dict_params
        
    def predict(self, X):
        c_X_batch=Chuy(array=X, calculate_grad=False, node_array=None)
        c_output_data=self.loss.predict(c_X_batch=c_X_batch)
        Y_predicted=self.loss.get_predicted_output(c_output_data=c_output_data)
        return Y_predicted

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
A CONTINUACION SE PROBARA TODO LO QUE SE HA HECHO PARE VERIFICAR QUE TODO ESTE CORRECTO
    - DIFERENTES TIPOS DE PROBLEMAS
        - LINEAR REGRESSION
        - LOGISTIC REGRESSION
        - NEURAL NETWORK
    - DIFERENTES FUNCIONES DE PERDIDA
    - DIFERENTES OPTIMIZADORES
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def plot_loss_history(history, optimizer_name):
    epochs=history.shape[0]
    fig,ax=plt.subplots(nrows=1, ncols=1)
    ax.plot(np.array(range(1,epochs+1)),history.flatten())
    plt.grid()
    ax.set_title("Progress (optimizer: {})".format(optimizer_name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss cost")
    fig.show()
    return fig

# """
# DATASET: housing_data.csv
# TYPE: LINEAR REGRESSION

# Datos:
#     - X (m,n)
#     - Y (m,1) (numeros reales)
# """

# df=pd.read_csv('./housing_data.csv', header=None, sep='\s+') 
# df.columns=["CRIM","ZN","INDUS" , "CHAS" ,"NOX","RM" ,"AGE","DIS", "RAD","TAX", "PTRATIO", "B" , "LSTAT" , "MEDV"]
# print(df.head())

# def plot_data(X_data, Y_data, model=None):
#     fig,ax=plt.subplots(nrows=1, ncols=1)
#     ax.scatter(x=X_data, y=Y_data)
#     if model is not None:
#         x_min,x_max=X_data[:,0].min(),X_data[:,0].max()
#         X_new=np.array([[x_min],[x_max]])
#         Y_predicted=model.predict(X=X_new)
#         ax.plot(X_new,Y_predicted)
#     plt.show()
    
# X=df['RM'].values[:,None]
# Y=df['MEDV'].values[:,None]

# # Se recomienda normalizar los datos (para un rapido entrenamiento)
# X_norm,X_mean,X_std=StandardNormalization.transform(data=X)
# Y_norm,Y_mean,Y_std=StandardNormalization.transform(data=Y)

# plot_data(X_data=X, Y_data=Y)
# plot_data(X_data=X_norm, Y_data=Y_norm)

# # Observaciones:
# #     - optimizer: mbgd
# #           - Con nesterov=True diverge xD             
# #           - Con nesterov=False funciona pero con un alpha muy pequeno   
# #     - optimizer: adagrad
# #       Converge   
# #     - optimizer: rmsprop
# #       Converge   
# #     - optimizer: adadelta
# #       Converge   
# #     - optimizer: adam
# #       Converge          

# _,n=X_norm.shape        # Numero de caracteristicas

# number_neurons_layer_0=n
# number_neurons_layer_1=1

# model=SequentialModel()
# layer_1=FCLayer(number_neurons=number_neurons_layer_1, activation_function_name="identity", input_size=number_neurons_layer_0)
# model.add(layer=layer_1)
# model.set_loss(loss_name="mse")
# # model.set_optimizer(optimizer_name="mbgd", alpha=0.0001, momentum=0.9, nesterov=False)
# # model.set_optimizer(optimizer_name="adagrad", alpha=1.0, eps=1e-8)
# # model.set_optimizer(optimizer_name="rmsprop", alpha=1.0, momentum=0.9, eps=1e-8)
# # model.set_optimizer(optimizer_name="adadelta", init_val_delta_param=0.1, momentum=0.9, eps=1e-8)
# model.set_optimizer(optimizer_name="adam", alpha=1.0, beta_1=0.9, beta_2=0.99, eps=1e-8)

# m=X_norm.shape[0]
# epochs=10
# history=model.fit(X=X_norm[:,[0]], Y=Y_norm[:,[0]], batch_size=m, epochs=epochs)
# loss_cost,_=model.evaluate(X_test=X_norm, Y_test=Y_norm)
# print("Loss cost: {}".format(loss_cost))

# fig=plot_loss_history(history=history, optimizer_name=model.optimizer_name)

# plot_data(X_data=X_norm, Y_data=Y_norm, model=model)

# """
# DATASET: logistic_regression_social_network_ads.csv
# TYPE: LOGISTIC REGRESSION

# Datos:
#     - X (m,n)
#     - Y (m,1) {0,1}
# """

# df=pd.read_csv('./logistic_regression_social_network_ads.csv', sep=',') 
# print(df.head())

# def limit_function(params, x2):
#     theta1,theta2,bias=params
#     x1=-(bias + theta2 * x2)/(theta1)
#     return x1

# def plot_data(X_data, Y_data, model=None):
#     color_0='blue'
#     color_1='red'
#     indexes_0=np.where(Y_data==0)[0]
#     indexes_1=np.where(Y_data==1)[0]
#     fig,ax=plt.subplots(nrows=1, ncols=1)
#     ax.scatter(x=X_data[indexes_0,0], y=X_data[indexes_0,1], color=color_0)
#     ax.scatter(x=X_data[indexes_1,0], y=X_data[indexes_1,1], color=color_1)
#     if model is not None:
#         ax.plot([limit_function(params=model.loss.get_param_vals().flatten(), x2=X_data[:,1].max()), limit_function(params=model.loss.get_param_vals().flatten(), x2=X_data[:,1].min())],[X_data[:,1].max(), X_data[:,1].min()])
#     plt.show()

# X=df[['Age','EstimatedSalary']].values
# Y=df[['Purchased']].values

# # Se recomienda normalizar los datos (para un rapido entrenamiento)
# X_norm,X_mean,X_std=StandardNormalization.transform(data=X)

# plot_data(X_data=X, Y_data=Y)
# plot_data(X_data=X_norm, Y_data=Y)

# # Observaciones:
# #     - optimizer: mbgd
# #           - Con nesterov=True converge           
# #           - Con nesterov=False converge  
# #     - optimizer: adagrad
# #       Converge   
# #     - optimizer: rmsprop
# #       Converge   
# #     - optimizer: adadelta
# #       Converge   
# #     - optimizer: adam
# #       Converge    

# _,n=X_norm.shape        # Numero de caracteristicas
# k=2                     # Numero de clases

# number_neurons_layer_0=n
# number_neurons_layer_1=1

# model=SequentialModel()
# layer_1=FCLayer(number_neurons=number_neurons_layer_1, activation_function_name="sigmoid", input_size=number_neurons_layer_0)
# model.add(layer=layer_1)
# model.set_loss(loss_name="bce")
# # model.set_optimizer(optimizer_name="mbgd", alpha=1.0, momentum=0.9, nesterov=False)
# # model.set_optimizer(optimizer_name="adagrad", alpha=1.0, eps=1e-8)
# # model.set_optimizer(optimizer_name="rmsprop", alpha=1.0, momentum=0.9, eps=1e-8)
# # model.set_optimizer(optimizer_name="adadelta", init_val_delta_param=0.1, momentum=0.9, eps=1e-8)
# model.set_optimizer(optimizer_name="adam", alpha=1.0, beta_1=0.9, beta_2=0.99, eps=1e-8)

# m=X_norm.shape[0]
# epochs=10
# history=model.fit(X=X_norm[:,:], Y=Y[:,[0]], batch_size=m, epochs=epochs)
# loss_cost,evaluation_metric=model.evaluate(X_test=X_norm, Y_test=Y, evaluation_metric_name="f1_score", number_classes=k)
# print("Confusion matrix:\n{}".format(model.evaluation_metric.confusion_matrix))
# print("Loss cost: {}".format(loss_cost))
# print("F1 score: {}".format(evaluation_metric.f1_score))

# fig=plot_loss_history(history=history, optimizer_name=model.optimizer_name)

# plot_data(X_data=X_norm, Y_data=Y, model=model)

# """
# DATASET: Iris.csv
# TYPE: NEURAL NETWORK

# Datos:
#     - X (m,n)
#     - Y (m,1) {0,1,2,...,k-1} (k: Numero de clases)
# """

# df=pd.read_csv('./Iris.csv', sep=',') 
# print(df.head())

# def plot_data(X_data, Y_data, model=None, n=20):
#     color_0='blue'
#     color_1='orange'
#     color_2='green'
#     indexes_0=np.where(Y_data==0)[0]
#     indexes_1=np.where(Y_data==1)[0]
#     indexes_2=np.where(Y_data==2)[0]
#     fig,ax=plt.subplots(nrows=1, ncols=1)
#     ax.scatter(x=X_data[indexes_0,0], y=X_data[indexes_0,1], color=color_0)
#     ax.scatter(x=X_data[indexes_1,0], y=X_data[indexes_1,1], color=color_1)
#     ax.scatter(x=X_data[indexes_2,0], y=X_data[indexes_2,1], color=color_2)
#     if model is not None:
#         min_x1,max_x1=X_data[:,0].min(),X_data[:,0].max()
#         min_x2,max_x2=X_data[:,1].min(),X_data[:,1].max()
#         vals_x1=np.linspace(min_x1,max_x1,n)
#         vals_x2=np.linspace(min_x2,max_x2,n)
#         l1,l2=len(vals_x1),len(vals_x2)
#         X_new=np.zeros((l1*l2,2))
#         pos=0
#         for i in range(l1):
#             for j in range(l2):
#                 X_new[pos,:]=np.array([vals_x1[i],vals_x2[j]])
#                 pos+=1
#         Y_predicted=model.predict(X=X_new)
#         pos=0
#         for i in range(l1):
#             for j in range(l2):
#                 y=Y_predicted[pos,0]
#                 color=color_0 if y == 0 else color_1 if y == 1 else color_2
#                 ax.scatter(x=vals_x1[i], y=vals_x2[j], color=color, marker='x', s=15)
#                 pos+=1
#     plt.show()
    
# # 'pandas.factorize'es un metodo bastante bueno para convertir categorias en indices 
# categories=np.array(df["Species"].factorize()[1])
# # array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)

# X=df[['SepalWidthCm','PetalLengthCm']].values
# Y=df["Species"].factorize()[0][:,None]

# # Se recomienda normalizar los datos (para un rapido entrenamiento)
# X_norm,X_mean,X_std=StandardNormalization.transform(data=X)

# plot_data(X_data=X, Y_data=Y)
# plot_data(X_data=X_norm, Y_data=Y)

# # Observaciones:
# #     - optimizer: mbgd
# #           - Con nesterov=True converge           
# #           - Con nesterov=False no lo hace bien  
# #     - optimizer: adagrad
# #       Converge   
# #     - optimizer: rmsprop
# #       Converge   
# #     - optimizer: adadelta
# #       No lo hizo tan bien   
# #     - optimizer: adam
# #       Converge  

# _,n=X_norm.shape        # Numero de caracteristicas
# k=3                     # Numero de clases

# number_neurons_layer_0=n
# number_neurons_layer_1=10
# number_neurons_layer_2=k

# model=SequentialModel()
# layer_1=FCLayer(number_neurons=number_neurons_layer_1, activation_function_name="sigmoid", input_size=number_neurons_layer_0)
# layer_2=FCLayer(number_neurons=number_neurons_layer_2, activation_function_name="softmax", input_size=number_neurons_layer_1)
# model.add(layer=layer_1)
# model.add(layer=layer_2)
# model.set_loss(loss_name="ce")
# # model.set_optimizer(optimizer_name="mbgd", alpha=1.0, momentum=0.9, nesterov=False)
# # model.set_optimizer(optimizer_name="adagrad", alpha=1.0, eps=1e-8)
# # model.set_optimizer(optimizer_name="rmsprop", alpha=1.0, momentum=0.9, eps=1e-8)
# # model.set_optimizer(optimizer_name="adadelta", init_val_delta_param=0.1, momentum=0.9, eps=1e-8)
# model.set_optimizer(optimizer_name="adam", alpha=1.0, beta_1=0.9, beta_2=0.99, eps=1e-8)

# m=X_norm.shape[0]
# epochs=10
# history=model.fit(X=X_norm[0:m,:], Y=Y[0:m,[0]], batch_size=m, epochs=epochs)
# loss_cost,evaluation_metric=model.evaluate(X_test=X_norm, Y_test=Y, evaluation_metric_name="f1_score", number_classes=k)
# print("Confusion matrix:\n{}".format(model.evaluation_metric.confusion_matrix))
# print("Loss cost: {}".format(loss_cost))
# print("F1 score: {}".format(evaluation_metric.f1_score))

# fig=plot_loss_history(history=history, optimizer_name=model.optimizer_name)

# plot_data(X_data=X_norm, Y_data=Y, model=model, n=25)

# """
# DATASET: XOR GATE PROBLEM
# TYPE: LOGISTIC REGRESSION - NEURAL NETWORK
# """

# def plot_data(X_data, Y_data, model=None, n=20):
#     color_0='blue'
#     color_1='red'
#     indexes_0=np.where(Y_data==0)[0]
#     indexes_1=np.where(Y_data==1)[0]
#     fig,ax=plt.subplots(nrows=1, ncols=1)
#     ax.scatter(x=X_data[indexes_0,0], y=X_data[indexes_0,1], color=color_0)
#     ax.scatter(x=X_data[indexes_1,0], y=X_data[indexes_1,1], color=color_1)
#     if model is not None:
#         pad=0.5
#         min_x1,max_x1=X_data[:,0].min() - pad,X_data[:,0].max() + pad
#         min_x2,max_x2=X_data[:,1].min() - pad,X_data[:,1].max() + pad
#         vals_x1=np.linspace(min_x1,max_x1,n)
#         vals_x2=np.linspace(min_x2,max_x2,n)
#         l1,l2=len(vals_x1),len(vals_x2)
#         X_new=np.zeros((l1*l2,4))
#         pos=0
#         for i in range(l1):
#             for j in range(l2):
#                 X_new[pos,:]=np.array([vals_x1[i],vals_x2[j],int(np.logical_not(vals_x1[i])),int(np.logical_not(vals_x2[j]))])
#                 pos+=1
#         Y_predicted=model.predict(X=X_new)
#         pos=0
#         for i in range(l1):
#             for j in range(l2):
#                 y=Y_predicted[pos,0]
#                 color=color_0 if y == 0 else color_1 
#                 ax.scatter(x=vals_x1[i], y=vals_x2[j], color=color, marker='x', s=15)
#                 pos+=1
#     plt.grid()
#     plt.show()
    
# X=np.array([[0,0,1,1],[0,1,1,0],[1,0,0,1],[1,1,0,0]])
# Y=np.array([[0],[1],[1],[0]])

# # Se recomienda normalizar los datos (para un rapido entrenamiento)
# X_norm,X_mean,X_std=StandardNormalization.transform(data=X)

# plot_data(X_data=X, Y_data=Y)
# plot_data(X_data=X_norm, Y_data=Y)

# # Observaciones:
# #     - optimizer: mbgd
# #           - Con nesterov=True converge           
# #           - Con nesterov=False no lo hace bien  
# #     - optimizer: adagrad
# #       Converge   
# #     - optimizer: rmsprop
# #       No lo hizo tan bien    
# #     - optimizer: adadelta
# #       No lo hizo tan bien   
# #     - optimizer: adam
# #       No lo hizo tan bien     

# _,n=X_norm.shape        # Numero de caracteristicas
# k=2                     # Numero de clases

# number_neurons_layer_0=n
# number_neurons_layer_1=2
# number_neurons_layer_2=1

# model=SequentialModel()
# layer_1=FCLayer(number_neurons=number_neurons_layer_1, activation_function_name="sigmoid", input_size=number_neurons_layer_0)
# layer_2=FCLayer(number_neurons=number_neurons_layer_2, activation_function_name="sigmoid", input_size=number_neurons_layer_1)
# model.add(layer=layer_1)
# model.add(layer=layer_2)
# model.set_loss(loss_name="bce")
# # model.set_optimizer(optimizer_name="mbgd", alpha=1.0, momentum=0.9, nesterov=False)
# # model.set_optimizer(optimizer_name="adagrad", alpha=1.0, eps=1e-8)
# # model.set_optimizer(optimizer_name="rmsprop", alpha=1.0, momentum=0.9, eps=1e-8)
# # model.set_optimizer(optimizer_name="adadelta", init_val_delta_param=0.1, momentum=0.9, eps=1e-8)
# model.set_optimizer(optimizer_name="adam", alpha=1.0, beta_1=0.9, beta_2=0.99, eps=1e-8)

# m=X_norm.shape[0]
# epochs=10
# history=model.fit(X=X_norm[0:m,:], Y=Y[0:m,[0]], batch_size=m, epochs=epochs)
# loss_cost,evaluation_metric=model.evaluate(X_test=X_norm, Y_test=Y, evaluation_metric_name="f1_score", number_classes=k)
# print("Confusion matrix:\n{}".format(model.evaluation_metric.confusion_matrix))
# print("Loss cost: {}".format(loss_cost))
# print("F1 score: {}".format(evaluation_metric.f1_score))

# fig=plot_loss_history(history=history, optimizer_name=model.optimizer_name)

# plot_data(X_data=X_norm, Y_data=Y, model=model, n=30)


























# """
# CREACION DE CLASIFICADORES
# """

# # NOTA: AQUI LOS DATOS YA SE DEBIERON DE HABER CAPTURADO

# # LIMPIEZA DE DATOS 

# for hand in ["left_hand", "right_hand"]:
#     path="/home/chuy/Practicas/Python/Project/UI18/game/dataset/{}".format(hand)
#     path_save="/home/chuy/Practicas/Python/Project/UI18/game/clean_dataset/{}".format(hand)
#     l=os.listdir(path)
#     l.sort()
#     local_euler_angles_closed_hand_name_list=[]
#     local_euler_angles_open_hand_name_list=[]
#     for i in range(len(l)):
#         name=l[i]
#         if name.split("_")[4] == "closed":
#             local_euler_angles_closed_hand_name_list.append(name)
#         else:
#             local_euler_angles_open_hand_name_list.append(name)

#     local_euler_angles_closed_hand_name_list
#     local_euler_angles_open_hand_name_list

#     local_euler_angles_closed_hand_list=[np.load("{}/{}".format(path,local_euler_angles_closed_hand_name_list[i])) for i in range(len(local_euler_angles_closed_hand_name_list))]
#     local_euler_angles_open_hand_list=[np.load("{}/{}".format(path,local_euler_angles_open_hand_name_list[i])) for i in range(len(local_euler_angles_open_hand_name_list))]

#     # Numeracion de dedos: 0 (pulgar,thumb), 1 (indice,index), 2 (medio,middle), 3 (anular,ring), 4 (menique,pinky)
#     # Lista de arrays de (3,3) (Numero de articulaciones,Numero de angulos por articulacion)
#     # Los ejemplos seran cada array desglosado (ejemplos de 9 caracteristicas)
#     m_closed_hand=len(local_euler_angles_closed_hand_list)
#     m_open_hand=len(local_euler_angles_open_hand_list)

#     thumb_closed_hand,thumb_open_hand=np.zeros((m_closed_hand,9)),np.zeros((m_open_hand,9))
#     index_closed_hand,index_open_hand=np.zeros((m_closed_hand,9)),np.zeros((m_open_hand,9))
#     middle_closed_hand,middle_open_hand=np.zeros((m_closed_hand,9)),np.zeros((m_open_hand,9))
#     ring_closed_hand,ring_open_hand=np.zeros((m_closed_hand,9)),np.zeros((m_open_hand,9))
#     pinky_closed_hand,pinky_open_hand=np.zeros((m_closed_hand,9)),np.zeros((m_open_hand,9))

#     for i in range(m_closed_hand):
#         array=local_euler_angles_closed_hand_list[i]
#         thumb_closed_hand[i,:]=array[0:3,:].flatten(order='C')
#         index_closed_hand[i,:]=array[3:6,:].flatten(order='C')
#         middle_closed_hand[i,:]=array[6:9,:].flatten(order='C')
#         ring_closed_hand[i,:]=array[9:12,:].flatten(order='C')
#         pinky_closed_hand[i,:]=array[12:15,:].flatten(order='C')

#     for i in range(m_open_hand):
#         array=local_euler_angles_open_hand_list[i]
#         thumb_open_hand[i,:]=array[0:3,:].flatten(order='C')
#         index_open_hand[i,:]=array[3:6,:].flatten(order='C')
#         middle_open_hand[i,:]=array[6:9,:].flatten(order='C')
#         ring_open_hand[i,:]=array[9:12,:].flatten(order='C')
#         pinky_open_hand[i,:]=array[12:15,:].flatten(order='C')

#     # Axis: 0 (a lo largo de las columnas), 1 (a lo largo de las filas)

#     aux_list=[
#         ("thumb_closed_hand", thumb_closed_hand),("thumb_open_hand", thumb_open_hand),
#         ("index_closed_hand", index_closed_hand),("index_open_hand", index_open_hand),
#         ("middle_closed_hand", middle_closed_hand),("middle_open_hand", middle_open_hand),
#         ("ring_closed_hand", ring_closed_hand),("ring_open_hand", ring_open_hand),
#         ("pinky_closed_hand", pinky_closed_hand),("pinky_open_hand", pinky_open_hand)
#     ]
    
#     # El orden es como el anterior
#     # (radio, numero seleccionados, numero total)
#     history=np.zeros((10,3))
#     count=0
#     for name,array in aux_list:
#         radius=""
#         option=""
#         while option != "f":
#             central_point=np.median(array, axis=0)
#             best_array_list=[]
            
#             radius=input("(hand: {}, type: {}) Ingrese un radio (en grados): ".format(hand,name)) 
#             radius=float(radius)
#             m,n=array.shape
#             for i in range(m):
#                 if np.linalg.norm(array[i] - central_point) <= radius ** 2:
#                     best_array_list.append(array[i])
#             option=input("Numero de elementos (con radio '{}'): '{}' (de un total de '{}'). Desea guardarlos? (si,no): ".format(radius, len(best_array_list), m))
#             if option == "si":
#                 k=len(best_array_list)
#                 best_array_dataset=np.zeros((k,9))
#                 for i in range(k):
#                     best_array=best_array_list[i]
#                     best_array_dataset[i,:]=best_array
#                 np.save("{}/best_{}_dataset.npy".format(path_save,name), best_array_dataset)
#                 history[count,:]=np.array([radius, k, m])
#                 count+=1
#                 option="f"
           
#     np.save("{}/history.npy".format(path_save), history)

# # CREACION DE LOS MODELOS DE CLASIFICACION

# for hand in ["left_hand", "right_hand"]:    
#     path="/home/chuy/Practicas/Python/Project/UI18/game/clean_dataset/{}".format(hand)
#     l=os.listdir(path)
    
#     X_list=[]
#     Y_list=[]
    
#     name_list=[
#         ("thumb_closed_hand", "thumb_open_hand"),
#         ("index_closed_hand", "index_open_hand"),
#         ("middle_closed_hand", "middle_open_hand"),
#         ("ring_closed_hand", "ring_open_hand"),
#         ("pinky_closed_hand", "pinky_open_hand")
#     ]
    
#     # Porcentajes para cada clase (closed, open)
#     train_p=0.70
#     test_p=1 - train_p
#     for name_closed,name_open in name_list:
#         array_closed=np.load("{}/best_{}_dataset.npy".format(path, name_closed))
#         array_open=np.load("{}/best_{}_dataset.npy".format(path, name_open))
    
#         n=9
#         m_closed=array_closed.shape[0]
#         m_open=array_open.shape[0]
    
#         closed_indexes=np.arange(0,m_closed)
#         train_closed_indexes=np.random.choice(closed_indexes,size=int(train_p * m_closed),replace=False)
#         test_closed_indexes=np.array(list(set(closed_indexes).difference(train_closed_indexes)))
#         X_closed_train=array_closed[train_closed_indexes,:]
#         X_closed_test=array_closed[test_closed_indexes,:]
    
#         open_indexes=np.arange(0,m_open)
#         train_open_indexes=np.random.choice(open_indexes,size=int(train_p * m_open),replace=False)
#         test_open_indexes=np.array(list(set(open_indexes).difference(train_open_indexes)))
#         X_open_train=array_open[train_open_indexes,:]
#         X_open_test=array_open[test_open_indexes,:]
        
#         X_train=np.concatenate([X_closed_train,X_open_train], axis=0)
#         Y_train=np.concatenate([np.zeros((X_closed_train.shape[0],1), dtype=int),np.ones((X_open_train.shape[0],1), dtype=int)], axis=0)
    
#         X_test=np.concatenate([X_closed_test,X_open_test], axis=0)
#         Y_test=np.concatenate([np.zeros((X_closed_test.shape[0],1), dtype=int),np.ones((X_open_test.shape[0],1), dtype=int)], axis=0)
    
#         # Para revolver los ejemplos
#         m=X_train.shape[0]
#         new_indexes=np.random.choice(np.arange(0,m),size=m,replace=False)
#         X_train=X_train[new_indexes,:]
#         Y_train=Y_train[new_indexes,:]
    
#         m=X_test.shape[0]
#         new_indexes=np.random.choice(np.arange(0,m),size=m,replace=False)
#         X_test=X_test[new_indexes,:]
#         Y_test=Y_test[new_indexes,:]
    
#         # Se recomienda normalizar los datos (para un rapido entrenamiento)
#         X_train_norm,X_train_mean,X_train_std=StandardNormalization.transform(data=X_train)
#         X_test_norm,_,_=StandardNormalization.transform(data=X_test, mean_std=(X_train_mean,X_train_std))
    
#         # CREACION DEL CLASIFICADOR
        
#         # Observaciones:
#         #     - 
        
#         m,n=X_train_norm.shape          # Numero de caracteristicas
#         k=2                             # Numero de clases
        
#         number_neurons_layer_0=n
#         number_neurons_layer_1=1
        
#         model=SequentialModel()
#         layer_1=FCLayer(number_neurons=number_neurons_layer_1, activation_function_name="sigmoid", input_size=number_neurons_layer_0)
#         model.add(layer=layer_1)
#         model.set_loss(loss_name="bce")
#         # model.set_optimizer(optimizer_name="mbgd", alpha=1.0, momentum=0.9, nesterov=True)
#         # model.set_optimizer(optimizer_name="adagrad", alpha=1.0, eps=1e-8)
#         # model.set_optimizer(optimizer_name="rmsprop", alpha=1.0, momentum=0.9, eps=1e-8)
#         # model.set_optimizer(optimizer_name="adadelta", init_val_delta_param=0.1, momentum=0.9, eps=1e-8)
#         model.set_optimizer(optimizer_name="adam", alpha=0.1, beta_1=0.9, beta_2=0.99, eps=1e-8)
        
#         epochs=10
#         history=model.fit(X=X_train_norm[:,:], Y=Y_train[:,[0]], batch_size=m, epochs=epochs)
#         loss_cost,evaluation_metric=model.evaluate(X_test=X_test_norm, Y_test=Y_test, evaluation_metric_name="f1_score", number_classes=k)
#         print("Confusion matrix:\n{}".format(model.evaluation_metric.confusion_matrix))
#         print("Loss cost: {}".format(loss_cost))
#         print("F1 score: {}".format(evaluation_metric.f1_score))
        
#         fig=plot_loss_history(history=history, optimizer_name=model.optimizer_name)
    
#         # GUARDAR PARAMETROS ENTRENADOOS
        
#         path_save_params="{}/models".format(path)
#         model_name="{}_{}".format(name_closed.split("_")[0], name_closed.split("_")[2])
        
#         model.save_params(path=path_save_params, name=model_name, other_configurations={
#             "normalization": (X_train_mean,X_train_std),
#             "epochs": epochs,
#             "history": history,
#             "loss_cost": loss_cost,
#             "f1_score": evaluation_metric.f1_score,
#             "optimizer_configurations": model.optimizer.get_dict_configurations()
#         })
        
#         fig.savefig("{}/{}_loss_history.png".format(path_save_params, model_name))
    
#         # CARGAR PARAMETROS ENTRENADOS
        
#         model=SequentialModel()
#         dict_params=model.load_params(path=path_save_params, name=model_name)
#         # Verificacion con los datos de entrenamiento
#         loss_cost,evaluation_metric=model.evaluate(X_test=X_train_norm, Y_test=Y_train, evaluation_metric_name="f1_score", number_classes=k)
#         print("Confusion matrix:\n{}".format(model.evaluation_metric.confusion_matrix))
#         print("Loss cost: {}".format(loss_cost))
#         print("F1 score: {}".format(evaluation_metric.f1_score))
        
#         # UTILIZAR MODELO CON NUEVOS DATOS
#         # NOTA: SE DEBEN NORMALIZAR CON LOS DATOS OBTENIDOS EN EL ENTRENAMIENTO (X_train_mean,X_train_std)
        
#         X_train_mean,X_train_std=dict_params["other_configurations"]["normalization"]
#         X_new=X_train[[0],:]
#         X_new_norm,_,_=StandardNormalization.transform(data=X_new, mean_std=(X_train_mean,X_train_std))
#         Y_predicted=model.predict(X=X_new_norm)
#         print("Real value: {}".format(Y_train[0,0]))
#         print("Predicted value: {}".format(Y_predicted[0,0]))
    











































