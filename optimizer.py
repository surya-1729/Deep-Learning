class SGD:
    
    def __init__(self, params, lr=1e-3):
        super().__init__()
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        # TODO: update the parameters' gradients according to
        #       gradient descent with step size lr;
        for par in self.params:
            par.data = par.data - self.lr*par.grad.data
    
    def zero_grad(self):
        # TODO: set all the parameters' gradients to zero
        #       by calling 'zero_' on the parameters' gradients
        for par in self.params:
            par.grad.data.zero_()