import pymc as pm



class bowtie(pm.Model): #extend pymc model
    '''Inheriting from the PyMC model to focus on the modelling of bowties


    https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html
    Causes are Inputs
    Consequences are Outputs
    Context is not a computational element
    Top Event is an observed variable only(predictive prior)
    Barriers are deterministic functions that have a threshold.  If the Input is higher than the threshold it will overcome the barrier.

    '''
    element_types = {'ca':'cause',
                    'co':'consequence',
                    'con':'context',
                    'te':'top event',
                    'pb':'preventative barrier',
                    'mb':'mitigation barrier',
                    'ef':'escalatory factor'}
    def __init__(self,name=''):
        super().__init__(name)
        self.causes=[]
        self.consequences=[]
        self.context=''
        self.topevent=None
        self.prevenativebarriers=[]
        self.mitigationbarriers=[]
        self.escalatoryfactors=[]
        self.variables=[]
   




def barrier(condition,input,threshold=0):
    return pm.math.switch(condition > threshold,input,0)

def topevent(input):
    return input

def factor(condition,input,threshold=0):
    return pm.math.switch(condition > threshold,input,0)

def cause(condition,input,threshold=0):
    return pm.math.switch(condition > threshold,input,0)

def consequence(condition,input,threshold=0):
    return pm.math.switch(condition > threshold,input,0)