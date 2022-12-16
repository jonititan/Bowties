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
if __name__ == '__main__': # necessary on windows only
    with bt.bowtie() as airprox:

        airprox.context = 'Flight activity'
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
        splitdist = pm.DiscreteUniform('Even Split',lower=0,upper=1)

        #Causes 
        cause1 = pm.Deterministic('Cause\nBad Luck',bt.cause(splitdist))
        cause2 = pm.Deterministic('Cause\nNo Flight Plan',bt.cause(bt.invert(splitdist)))
        airprox.causes = ['Cause\nBad Luck','Cause\nNo Flight Plan']
            

        #Preventory Barriers
        barrier1 = pm.Deterministic('Barrier\nFlight Plan',bt.barrier(c,cause1))
        barrier2 = pm.Deterministic('Barrier\nATC Warning',bt.barrier(d,bt.combine((cause2,barrier1))))
        barrier3 = pm.Deterministic('Barrier\nPilot Sky\nScanning',bt.barrier(d,barrier2,threshold=2))
        airprox.prevenativebarriers = ['Barrier\nFlight Plan','Barrier\nATC Warning','Barrier\nPilot Sky\nScanning']
        
        #Top Event
        te = pm.Deterministic('Top Event\nImminent loss\nof seperation',bt.topevent(barrier3))
        airprox.topevent = 'Top Event\nImminent loss\nof seperation'
        
        #Mitigation Barriers
        barrier4 = pm.Deterministic('Barrier\nSituational Awareness',bt.barrier(e,te))
        airprox.mitigationbarriers = ['Barrier\nSituational Awareness']
        
        #Consequences
        consequence1 = pm.Deterministic('Consequence\nCollision',bt.consequence(barrier4))
        consequence2 = pm.Deterministic('Consequence\nAvoiding Action',bt.consequence(bt.inverting_and(barrier4,te)))
        airprox.consequences = ['Consequence\nCollision','Consequence\nAvoiding Action']
        trace = pm.sample(draws=nb_samples, random_seed=1000)


    print(airprox.allbarriers())
    print(list(airprox.named_vars.keys()))
    # ap_effectiveness = airprox.barrier_effectiveness(trace)


    bowtieplot = bt.plot_all_elements(airprox)
    bowtieplot.view()
    bowtienx = airprox.plot_bowtie(trace,
                                    e2r=[('Top Event\nImminent loss\nof seperation','Consequence\nAvoiding Action')]) #removing extra edge caused by inverting_and function

    bowtienx.view()

    model_nodes = list(airprox.named_vars.keys())
    RVNodes = ['a','b','c','d','e','f','g']
    DeterministicNodes = [a for a in model_nodes if a not in RVNodes]

    az.rcParams["plot.max_subplots"]=100
    az.plot_trace(trace,RVNodes)
    plt.suptitle('Airprox RV Nodes')
    plt.tight_layout()
    plt.show()

    az.rcParams["plot.max_subplots"]=100
    az.plot_trace(trace,DeterministicNodes)
    plt.suptitle('Airprox Deterministic Nodes')
    plt.tight_layout()
    plt.show()

    import code
    code.interact(local=locals())
