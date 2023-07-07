import numpy as np

def simple_mock_lc(fq, tmin, tmax, dnum = 1000):
    
    
    t = np.linspace(tmin, tmax, dnum, endpoint=False)
    x = np.sin(f * (2*np.pi) * t)
    
    return t, x