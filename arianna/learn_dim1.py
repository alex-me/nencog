# coding: utf-8
# # Nengo Learning Example: impara una funzione ad un solo parametro,  f(x)
'''
Questo file presenta un esempio di rete neurale, implementata con Nengo, in grado di
apprendere una funzione ad un singolo parametro, della forma f(x).
Viene definita una fase di addestramento preliminare in cui la rete apprende a calcolare la funzione.
'''

import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.processes import WhiteSignal


# definizione di parametri globali utilizzati dalla rete
global dset  # dset rappresenta l'insieme dei dati di input passati alla rete, utilizzato sia in fase di apprendimento che di recall
global label # label rappresenta l'insieme dei risultati effettivi calcolati sugli input in dset : label[i] = f(dset[i])
dset = [] 
label = [] 
global A_funct_probe, input_node_probe, output_node_probe, A_funct_probe, error_probe, learn_probe

# funzione che costruisce dset e label per l'addestramento della rete
def build_dset( fun,        # la funzione su cui si calcola il dataset
                ymin=-1,    # il valore più piccolo del dominio
                ymax=1,     # il valore più grande del dominio 
                dim=1,      # il numero di parametri della funzione (la dimensione del vettore di input)
                size=100    # il numero di elementi del dataset
            ):
    for i in range(size+1):
        inp = []
        for j in range(dim): 
            inp.append(np.random.random()*(np.abs(ymax-ymin))+ymin)
        dset.append(inp)
        label.append(fun(inp))
    return [dset,label]



# funzione per la creazione della rete neurale
def setup_nn(funct,         # la funzione che deve essere rappresentata
            n_neurons=100,  # numero di neuroni nella ensamble
            training=1,     # vale 1 se la rete deve essere addestrata sui valori di dataset, 0 altrimenti
            learning_rate=3e-4,
            nn = None       # rete neurale ottenuta dal pre-training
            ):
    
    model = nengo.Network()
    with model:
        model.training = training
        # si tratta di una rete basata su spiking neurons il cui tempo di reazione è un valore maggiore di 0
        # per questo motivo ogni segnale di input viene fornito per un tempo lungo 1 decimo di secondo
        input_node = nengo.Node(output=lambda t: dset[int(t*10)])
        output_node = nengo.Node(output=lambda t: label[int(t*10)])
        A = nengo.Ensemble(n_neurons, dimensions=1)
        A_funct = nengo.Ensemble(n_neurons, dimensions=1, radius=2.0)

        nengo.Connection(input_node, A)
        conn = nengo.Connection(A, A_funct)
        # Apply the PES learning rule to conn
        conn.learning_rule_type = nengo.PES(learning_rate)

        error = nengo.Ensemble(n_neurons, dimensions=1)
        # Provide an error signal to the learning rule
        if training:        
            nengo.Connection(error, conn.learning_rule)
        # Compute the error signal (error = actual - target)
        nengo.Connection(A_funct, error)
        nengo.Connection(output_node, error, transform=-1)
    
        stop_learning = nengo.Node(output=lambda t: not model.training)
        nengo.Connection(
            stop_learning,
            error.neurons,
            transform=-20 * np.ones((error.n_neurons, 1)))

    # variabili utilizzate per il successivo plotting dei risultati attraverso dei grafici
    with model:
        global A_funct_probe, input_node_probe, output_node_probe, A_funct_probe, error_probe, learn_probe
        input_node_probe = nengo.Probe(input_node)
        output_node_probe = nengo.Probe(output_node)
        A_probe = nengo.Probe(A, synapse=0.01)
        A_funct_probe = nengo.Probe(A_funct, synapse=0.01)
        error_probe = nengo.Probe(error, synapse=0.01)
        learn_probe = nengo.Probe(stop_learning, synapse=None)
    return model


# simulazione del modello (model) per un durata pari a (time)
# e' anche possibile riprendere la simulazione di un modello precedentemente interrotta, passando il simulatore (simulator)
def run_nn(simulator=None, model=None, time=10):
    sim = simulator
    if sim==None:
        sim = nengo.Simulator(model)
    sim.run(time)
    return [model,sim]


# funzione per valutare la risposta della rete alla funzione (funct) per un determinato input (xinput)
# la funzione prende in input il simulatore di una rete precedentemente addestrata (simulator)
# in base al tempo (t) di esecuzione del simulatore, viene modificato il flusso del segnale di input (deset) in modo
# da passare alla rete i segnali per cui si vuole calcolaore la risposta (xinput)
# l'esecuzione della rete viene protratta per un secondo in modo da stabilizzare il segnale di output
# viene restituito l'ultimo segnale di output prodotto dalla rete
def recall_nn(simulator, xinput, funct):
    if simulator==None:
        print "Erroe: Fornire un simulatore in input!"
    t = simulator.time
    for i in range(10):  # definiamo un segnale della lunghezza di un secondo
        dset[int(t*10)+i] = xinput
        label[int(t*10)+i] = funct(xinput)
    run_data = run_nn(simulator=simulator, time=1)
    model = run_data[0]
    sim = run_data[1]
    t = simulator.time # tempo dell'ultimo segnale
    return sim.data[A_funct_probe][int(t)*1000-1] # utlimo segnale

# Plot dei dati di simulazione
def plot_sim(sim):
    plt.figure(figsize=(9, 9))
    plt.subplot(3, 1, 1)
    plt.plot(
        sim.trange(),
        sim.data[input_node_probe],
        label='Input',
        color='k',
        linewidth=2.0)
    plt.plot(
        sim.trange(),
        sim.data[learn_probe],
        label='Stop learning?',
        color='r',
        linewidth=2.0)
    plt.legend(loc='lower right')
    #plt.ylim(-1.2, 1.2)

    plt.subplot(3, 1, 2)
    plt.plot(
        sim.trange(),
        sim.data[output_node_probe],
        label='Perfect result',
        linewidth=2.0)
    plt.plot(
        sim.trange(),
        sim.data[A_funct_probe],
        label='Decoded Ensemble')
    plt.legend(loc='lower right')
    #plt.ylim(-1.2, 1.2)

    plt.subplot(3, 1, 3)
    plt.plot(
        sim.trange(),
        sim.data[A_funct_probe] - sim.data[output_node_probe],
        label = 'Error')
    plt.legend(loc='lower right')
    
    plt.tight_layout();

    plt.show()



# esempio di esecuzione

def funct( x ):
    # questa e` la funzione che si vorrebbe far apprendere
    #return np.sin( np.pi * x[0] )
    return np.sin(np.pi*x[0])-np.cos(np.pi*x[0])

# Fase di Training della rete
training_time = 40 # tempo in secondi di addestramento (equivale a 400 input/10 al secondo)
dataset = build_dset(funct,size=10000) 
model   = setup_nn(funct,training=1) # training=1 sta per l'addestramento della rete
run_data = run_nn(model=model,time=training_time)
trained_model   = run_data[0]
sim     = run_data[1]
trained_model.training = 0;  # fase di training terminata


# fase di verifica:
# si propongono 10 (attempts) valori di input e si calcola l'errore medio rispetto ai risultati effettivi
mean_error = 0.0
attempts = 10
for i in range(attempts):
    x = np.random.random()*2-1.0  
    r = recall_nn(sim, [x], funct)
    v = funct([x])
    print "nn:",round(r,2), " real:", round(v,2)
    mean_error += np.abs(r-v)
print "mean-error: ", mean_error/attempts


# stampiamo un grafico della simulazione (training + verifica)
plot_sim(sim)
