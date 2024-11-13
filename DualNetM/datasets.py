import pandas as pd


def load_human_prior_network():
    prior_net = pd.read_csv('./data/NicheNet_human.csv', index_col=None, header=0)
    return prior_net

def load_mouse_prior_network():
    prior_net = pd.read_csv('./data/network_mouse.csv', index_col=None, header=0)
    return prior_net


