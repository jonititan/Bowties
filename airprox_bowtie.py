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
    # EventSignal = pm.ConstantData('True',1)
    # NoEventSignal = pm.ConstantData('False',0)
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    c = pm.Normal("c", mu=0.0, sigma=5.0)
    d = pm.Normal("d", mu=0.0, sigma=1.0)
    e = pm.Normal("e", mu=0.0, sigma=5.0)
    f = pm.Normal("f", mu=0.0, sigma=5.0)
    g = pm.Normal("g", mu=0.0, sigma=5.0)
    #Causes
    cause1 = pm.Deterministic('Cause\nBad Luck',bt.cause(a))
    cause2 = pm.Deterministic('Cause\nNo Flight Plan',bt.cause(b))
        #binary 

    #Preventory Barriers
    barrier1 = pm.Deterministic('Barrier\nFlight Plan',bt.barrier(c,cause1))
    barrier2 = pm.Deterministic('Barrier\nATC Warning',bt.barrier(d,bt.combine((cause2,barrier1))))
    #Top Event
    te = pm.Deterministic('Top Event\nImminent loss of seperation',bt.topevent(barrier2))

    #Mitigation Barriers
    barrier3 = pm.Deterministic('Barrier\nSituational Awareness',bt.barrier(e,te))

    #Consequences
    consequence1 = pm.Deterministic('Consequence\nCollision',bt.consequence(f,barrier3))
    consequence2 = pm.Deterministic('Consequence\nAvoiding Action',bt.consequence(g,barrier3))

    trace = pm.sample(draws=nb_samples, random_seed=1000)

bowtieplot = bt.plot_all_elements(airprox)
bowtieplot.view()
bowtienx = bt.plot_bowtie(airprox)
bowtienx.view()






model_nodes = list(airprox.named_vars.keys())

az.rcParams["plot.max_subplots"]=100
az.plot_trace(trace,model_nodes)
plt.title('Airprox Nodes')
plt.tight_layout()
plt.show()

import code
code.interact(local=locals())
