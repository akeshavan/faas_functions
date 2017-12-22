import numpy as np
import simplejson as json

def get_arr(n,m):
    return np.zeros((n,m)).tolist()

def handle(st):
    args = json.loads(st)
    out = get_arr(**args)
    print(out)
