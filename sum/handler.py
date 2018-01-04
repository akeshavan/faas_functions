import simplejson as json
import numpy as np

def doSum(n):
    return np.sum(np.arange(n))    

def handle(st):
    inp = json.loads(st)
    outp = doSum(**inp)
    print(outp)
