import pymc as pm
import networkx as nx
import arviz as az

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

def nx_to_gv(nx_graph):
    import graphviz
    graph = graphviz.Digraph(nx_graph.name)
    for node in nx_graph.nodes(data=True):
        graph.node(node[0],**node[1])
    graph.edges(nx_graph.edges())
    return graph

def plot_all_elements(model):
    mcgraph = pm.model_to_graphviz(model)
    mcgraph.attr(rankdir='LR')
    return mcgraph

def plot_bowtie(model,trace):
    nxgraph = pm.model_to_networkx(model)
    effective = barrier_effectiveness(model,trace)
    mapping = {}
    for node in nxgraph.nodes(data=True):
        node[1]['label'] = node[1]['label'].split('~')[0] + '{:.2f}'.format(effective[node[0]])
        mapping[node[0]] = node[1]
    nx.set_node_attributes(nxgraph,mapping)
    remove = [node[0] for node in nxgraph.nodes(data=True) if node[1]['shape']=='ellipse']
    nxgraph.remove_nodes_from(remove)
    mcgraph = nx_to_gv(nxgraph)
    mcgraph.attr(rankdir='LR')
    return mcgraph

def combine(inputList):
    return pm.math.switch(pm.math.sum(inputList) >= 1,1,0)


def barrier(condition,input,threshold=1):
    return pm.math.switch(condition > threshold,input,0)
  

def topevent(input):
    return input

def factor(condition,input,threshold=0):
    return pm.math.switch(condition > threshold,input,0)

def cause(condition,input=1,threshold=0):
    return pm.math.switch(condition > threshold,input,0)

def consequence(condition,input=1,threshold=0):
    return pm.math.switch(condition > threshold,input,0)

def barrier_effectiveness(model,trace):
    #the current implementation only calculates the probability of an event getting past that node.
    #not the effectiveness of the barrier as a function of how many signals at 1 got to the node
    model_nodes = list(model.named_vars.keys())
    td = az.convert_to_dataset(trace)
    effectiveness={}
    for node in model_nodes:
        no_samples = td[node].data.shape[0] * td[node].data.shape[1]
        effectiveness[node] = 1 - (td[node].data.sum()/no_samples)
    return effectiveness

