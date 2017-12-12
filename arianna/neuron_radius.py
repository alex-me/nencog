# Questo esempio dimostra l'utilizzo del parametro radius in un singolo neurone
# radius indica il valore (ampiezza) massimo di firing del neurone

import nengo
import numpy as np

model = nengo.Network()
with model:
    A = nengo.Ensemble(n_neurons=1, dimensions=1)
    A.intercepts=nengo.dists.Uniform(0, 0)
    A.radius = 0.5 # questo neurone spara fino a 0.5
    A.max_rates=nengo.dists.Uniform(100, 100)
    A.encoders=[[1]] #firing per valori positivi

    B = nengo.Ensemble(n_neurons=1, dimensions=1)
    B.intercepts=nengo.dists.Uniform(0, 0)
    B.radius = 2.3 # questo neurone spara fino a 2.3
    B.max_rates=nengo.dists.Uniform(100, 100)
    B.encoders=[[1]] #firing per valori positivi

    stim = nengo.Node(lambda t: np.cos(8 * t) * np.sin(t))

    nengo.Connection(stim, A)    
    nengo.Connection(stim, B)