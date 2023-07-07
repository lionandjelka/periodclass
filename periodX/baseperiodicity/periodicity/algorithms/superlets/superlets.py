# Superlet package included in this package
# Reference: Implementation by Gregor Mönke: github.com/tensionhead
from .superlet import superlet
from .superlet import gen_superlet_testdata, scale_from_period

import numpy as np
from periodicity.utils.correlation import correlation_nd

def superlets_methods(tt, mag, ntau, minfq = 10, maxfq = 500):
    """Perform hybrid2d method using superlets

        Parameters
        ----------
        tt : list of time data
        mag : list of magnitude values
        ntau, ngrid : values for controling wwz execution (see inp_param function)
        minfq : minimum frequency
        maxfq : maximum fraquency

    """
    
    mx = (tt.max()-tt.min())/2.
    mn = np.min(np.diff(tt))
    fmin = 1./minfq
    fmax = 1./maxfq
    df = (fmax-fmin) / ntau
    flist=np.arange(fmin, fmax + df, df)
    
    print(scale_from_period(1/flist))
    # Superlet calculation
    gg=superlet(
        mag,
        samplerate=1/np.diff(tt).mean(),
        scales=scale_from_period(1/flist),
        order_max=100,
        order_min = 1,
        c_1=3,
        adaptive=True,
    )
    
    # Hybrid 2D method
    gg1=np.abs(gg)
    corr = correlation_nd(gg1, gg1)
    extentmin=np.min(corr)
    extentmax=np.max(corr)

    extent=[extentmin,extentmax,extentmin,extentmax]
    

    return  corr,  extent
