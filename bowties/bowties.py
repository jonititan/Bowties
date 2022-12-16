import pymc as pm
import networkx as nx
import arviz as az
import numpy as np

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
        self.consequencesum=None
        self.context=''
        self.topevent=''
        self.prevenativebarriers=[]
        self.mitigationbarriers=[]
        self.escalatoryfactors=[]
        self.variables=[]

    def setstyles(self):
        self.styles = {'ca':{'fillcolor':'blue','shape':'box','style':'filled','fontcolor':'white'},
                        'co':{'fillcolor':'red','shape':'box','style':'filled',},
                        'con':{'label':self.context},
                        'te':{'fillcolor':'red','shape':'circle','style':'filled',},
                        'pb':{'fillcolor':'white','shape':'box','style':'filled',},
                        'mb':{'fillcolor':'white','shape':'box','style':'filled',},
                        'ef':{'fillcolor':'yellow','shape':'box','style':'filled',}}
    
    def allbarriers(self):
        all = []
        all.extend(self.prevenativebarriers)
        all.extend(self.mitigationbarriers)
        return all
    
    def finalnodes(self):
        all = []
        all.extend(self.consequences)
        all.append(self.topevent)
        return all

    
    
    def allbowtie(self):
        all = []
        all.extend(self.prevenativebarriers)
        all.extend(self.mitigationbarriers)
        all.append(self.topevent)
        all.append(self.context)
        all.extend(self.causes)
        all.extend(self.escalatoryfactors)
        all.extend(self.consequences)
        return all

    def barrier_effectiveness(self,trace):
        nxgraph = pm.model_to_networkx(self)
        td = az.convert_to_dataset(trace)
        effectiveness={}
        for node in self.allbarriers():
            nodeparents = list(nx.bfs_edges(nxgraph,source=node,depth_limit=1,reverse=True))
            if not len(nodeparents) >= 1:
                raise AttributeError('Barriers must have at least one parent node')
            else:
                parenttotal = 0
                for pars in nodeparents:
                    if pars[1] in self.allbowtie():
                        parenttotal+=td[pars[1]].data.sum()
            effectiveness[node] = 1 - (td[node].data.sum()/parenttotal)
        return effectiveness

    def cumulative_barrier_effectiveness(self,trace):
        td = az.convert_to_dataset(trace)
        effectiveness={}
        for node in self.allbarriers():
            no_samples = td[node].data.shape[0] * td[node].data.shape[1]
            effectiveness[node] = 1 - (td[node].data.sum()/no_samples)
        return effectiveness
    
    def consequence_likelihood(self,trace):
        td = az.convert_to_dataset(trace)
        likelihood={}
        for node in self.finalnodes():
            if node == self.topevent:
                no_samples = td[node].data.shape[0] * td[node].data.shape[1]
                likelihood[node] = 1-(td[node].data.sum()/no_samples)
            else:
                no_samples = td[node].data.shape[0] * td[node].data.shape[1]
                likelihood[node] = (td[node].data.sum()/no_samples)
        self.consequencesum = sum(likelihood.values())
        return likelihood

    def cause_likelihood(self,trace):
        td = az.convert_to_dataset(trace)
        likelihood={}
        for node in self.causes:
            no_samples = td[node].data.shape[0] * td[node].data.shape[1]
            likelihood[node] = (td[node].data.sum()/no_samples)
        return likelihood
    
    def plot_bowtie(self,trace,e2r=None):
        nxgraph = pm.model_to_networkx(self)
        effective = self.barrier_effectiveness(trace)
        cumulativeeffective = self.cumulative_barrier_effectiveness(trace)
        consequences = self.consequence_likelihood(trace)
        causes = self.cause_likelihood(trace)
        self.setstyles()
        mapping = {}
        for node in nxgraph.nodes(data=True):
            if node[0] in self.allbarriers():
                node[1]['label'] = node[1]['label'].split('~')[0] + 'E: {:.4f} CE: {:.4f}'.format(effective[node[0]],cumulativeeffective[node[0]])
                mapping[node[0]] = node[1]
                if node[0] in self.prevenativebarriers:
                    for key,value in self.styles['pb'].items():
                       mapping[node[0]][key] = value
                elif node[0] in self.mitigationbarriers:
                    for key,value in self.styles['mb'].items():
                       mapping[node[0]][key] = value

            elif node[0] in self.consequences:
                node[1]['label'] = node[1]['label'].split('~')[0] + '{:.4f}'.format(consequences[node[0]])
                mapping[node[0]] = node[1]
                for key,value in self.styles['co'].items():
                    mapping[node[0]][key] = value
            elif node[0] == self.topevent:
                node[1]['label'] = node[1]['label'].split('~')[0] + '\nProbability of\nNo Consequences\n{:.4f}'.format(consequences[node[0]])
                mapping[node[0]] = node[1]
                for key,value in self.styles['te'].items():
                       mapping[node[0]][key] = value
            elif node[0] in self.causes: 
                node[1]['label'] = node[1]['label'].split('~')[0] + '{:.4f}'.format(causes[node[0]])
                mapping[node[0]] = node[1]
                for key,value in self.styles['ca'].items():
                    mapping[node[0]][key] = value     
            else:
                node[1]['label'] = node[1]['label'].split('~')[0]
                mapping[node[0]] = node[1]
                if node[0] == self.context:
                    for key,value in self.styles['con'].items():
                       mapping[node[0]][key] = value
                elif node[0] in self.escalatoryfactors:
                    for key,value in self.styles['ef'].items():
                       mapping[node[0]][key] = value
        
        nx.set_node_attributes(nxgraph,mapping)
        remove = [node[0] for node in nxgraph.nodes(data=True) if node[0] not in self.allbowtie()]
        nxgraph.remove_nodes_from(remove)
        if not e2r is None:
            nxgraph.remove_edges_from(e2r)
        mcgraph = nx_to_gv(nxgraph)
        mcgraph.attr(rankdir='LR')
        return mcgraph

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


def combine(inputList):
    return pm.math.switch(pm.math.sum(inputList) >= 1,1,0)

def invert(input):
    return pm.math.switch(pm.math.sum(input) >= 1,0,1)

def inverting_and(input,test):
    # if test True then invert.  Otherwise pass through as supplied
    return pm.math.switch(pm.math.sum(test) >= 1,invert(input),input)

def barrier(condition,input,threshold=1):
    return pm.math.switch(condition > threshold,input,0)
  

def topevent(input):
    return input

def factor(condition,input,threshold=0):
    return pm.math.switch(condition > threshold,input,0)

def cause(condition,input=1,threshold=0):
    return pm.math.switch(condition > threshold,input,0)

def consequence(input):
    return input



