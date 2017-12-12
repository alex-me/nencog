# Questo esempio dimostra l'utilizzo del parametro max_rates in un singolo neurone
# max_rates indica la ferquenza massima di firing del neurone

import nengo
import numpy as np

model = nengo.Network()
with model:
    A = nengo.Ensemble(n_neurons=1, dimensions=1)
    A.intercepts=nengo.dists.Uniform(-.5, -.5)
    A.max_rates=nengo.dists.Uniform(200, 200)
    A.encoders=[[1]] #firing per valori positivi

    B = nengo.Ensemble(n_neurons=1, dimensions=1)
    B.intercepts=nengo.dists.Uniform(-.5, -.5)
    B.max_rates=nengo.dists.Uniform(50, 50)
    B.encoders=[[1]] #firing per valori positivi

    # Nodo che rappresenta il segnale di input
    stim = nengo.Node(lambda t: np.cos(8 * t) * np.sin(t))

    nengo.Connection(stim, A)    
    nengo.Connection(stim, B)