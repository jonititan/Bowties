import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import aesara
import aesara.tensor as at
from scipy import stats
import numpy as np
import os
import time
import bowties as bt

print(
    f"""
# Aesara version: {aesara.__version__}
# PyMC version: {pm.__version__}
"""
)

aesara.config.change_flags(exception_verbosity="high",optimizer=None)

nb_samples=2000
with bt.bowtie() as airprox:
    #variables

    a = pm.Normal("a", mu=0.0, sigma=5.0)
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    # a = pm.Normal("a", mu=0.0, sigma=5.0)
    # b = pm.Normal("b", mu=0.0, sigma=1.0)
    # a = pm.Normal("a", mu=0.0, sigma=5.0)
    # b = pm.Normal("b", mu=0.0, sigma=1.0)
    # a = pm.Normal("a", mu=0.0, sigma=5.0)
    # b = pm.Normal("b", mu=0.0, sigma=1.0)
    #Causes
    cause1 = pm.Deterministic('Cause: Bad Luck',bt.cause(a,a))
    # cause1 = pm.Deterministic('Bad Luck',bt.cause(a,a))
        #binary 

    #Preventory Barriers
    barrier1 = pm.Deterministic('Barrier: Flight Plan',bt.barrier(a,cause1))
    barrier2 = pm.Deterministic('Barrier: ATC Warning',bt.barrier(a,barrier1))
    #Top Event
    te = pm.Deterministic('Top Event: Imminent loss of seperation',bt.topevent(barrier2))

    #Mitigation Barriers
    barrier3 = pm.Deterministic('Barrier: Situational Awareness',bt.barrier(b,te))

    #Consequences
    consequence1 = pm.Deterministic('Consequence: Collision',bt.consequence(a,barrier3))
    consequence2 = pm.Deterministic('Consequence: Avoiding Action',bt.consequence(b,barrier3))

    trace = pm.sample(draws=nb_samples, random_seed=1000)

mcgraph = pm.model_to_graphviz(airprox)

mcgraph.view()
model_nodes = list(airprox.named_vars.keys())
az.rcParams["plot.max_subplots"]=100
az.plot_trace(trace,model_nodes)
plt.title('Airprox Nodes')
plt.tight_layout()
plt.show()

# import code
# code.interact(local=locals())
