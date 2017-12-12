# Questo esempio dimostra l'utilizzo del parametro intercetps in un singolo neurone
# intercepts rappresenta il valore al disopra del quale il neurone inizia la sua attivita di firing

import nengo
import numpy as np

model = nengo.Network()
with model:
    A = nengo.Ensemble(n_neurons=1, dimensions=1)
    # firing solo per valori superiori a 0.5
    A.intercepts=nengo.dists.Uniform(.5, .5)
    A.max_rates=nengo.dists.Uniform(100, 100)
    A.encoders=[[1]] #firing per valori positivi

    B = nengo.Ensemble(n_neurons=1, dimensions=1)
    # firing solo per valori superiori a -0.5
    B.intercepts=nengo.dists.Uniform(-.5, -.5)
    B.max_rates=nengo.dists.Uniform(100, 100)
    B.encoders=[[1]] #firing per valori positivi

    stim = nengo.Node(lambda t: np.cos(8 * t) * np.sin(t))

    nengo.Connection(stim, A)    
    nengo.Connection(stim, B)

    