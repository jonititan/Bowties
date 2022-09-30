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
def barrier(condition,input,threshold=0):
    return pm.math.switch(condition,input,0)

nb_samples=2000
with pm.Model() as model:
    EventSignal = pm.ConstantData('Signal',1)
    a = pm.Normal("a", mu=3.0, sigma=5.0)
    barrier2 = pm.Deterministic('Barrier: ATC Warning',barrier(a,EventSignal))
    trace = pm.sample(draws=nb_samples, random_seed=1000)

model_nodes = list(model.named_vars.keys())
az.rcParams["plot.max_subplots"]=100
az.plot_trace(trace,model_nodes)
plt.title('Airprox Nodes')
plt.tight_layout()
plt.show()