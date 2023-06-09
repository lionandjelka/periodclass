import numpy as np
import json
from scipy.signal import chirp, find_peaks, peak_widths
from traitlets.traitlets import Integer
from scipy import interpolate, optimize
from scipy.stats.mstats import mquantiles
from sklearn.utils import shuffle
from periodicity.utils.correlation import correlation_nd
from periodicity.algorithms.wavelets import *


def peak_interp(corr, ngrid,  minfq = 500, maxfq = 10):
    """Interpolation of the peak
        Parameters
        ----------
     
        corr : hybrid 2d output
        minfq : minimum frequency
        maxfq : maximum fraquency
    """
    
    hh1=np.rot90(corr).T/np.rot90(corr).T.max()
    hh1arr=np.rot90(hh1.T)
    hh1arr1=np.abs(hh1arr).sum(1)/np.abs(hh1arr).sum(1).max()

    fmin =1/minfq
    fmax = 1/maxfq
    df = (fmax - fmin) / ngrid
    
    # osax interpolation (stacked data of h2d along one axis) to obtain more points
    osax=np.arange(start=fmin,stop=fmax+df,step=df)
    xax =np.arange(start=fmin, stop = fmax + df, step = df/2)
    f = interpolate.interp1d(osax, np.abs(hh1arr1), fill_value="extrapolate")
    yax = []
    for v in xax:
        yax.append(float(f(v)))
    yax = np.array(yax)
    return yax

def signifGrid_john(numlc, peak, corr,  tt, yy, ntau,ngrid, f = 2, peakHeight = 0.6, minfq = 500, maxfq = 10, algorithm ='wwz', method = 'linear'):
    """Determination of significance usign Johnson method but for Grid search of parameters

        Parameters
        ----------
        numlc : int, number of lc for determination
        peak : determined periodicity peak
        corr : hybrid 2d output
        tt : time
        yy: magnitude
        plot: True of Folse
        ntau, ngrid, f : values for controling wwz execution (see inp_param function)
        minfq : minimum frequency
        maxfq : maximum fraquency
        peakHeight: max peak height,
        algorithm: wwz or superlets (for now)
    """
    
   # hh1=np.rot90(corr).T/np.rot90(corr).T.max()
   # hh1arr=np.rot90(hh1.T)
   # hh1arr1=np.abs(hh1arr).sum(1)/np.abs(hh1arr).sum(1).max()

   # fmin =1/minfq
   # fmax = 1/maxfq
   # df = (fmax - fmin) / ngrid
    
    # osax interpolation (stacked data of h2d along one axis) to obtain more points
    #osax=np.arange(start=fmin,stop=fmax+df,step=df)
    #xax =np.arange(start=fmin, stop = fmax + df, step = df/2)
    #f = interpolate.interp1d(osax, np.abs(hh1arr1), fill_value="extrapolate")
    #yax = []
    #for v in xax:
    #    yax.append(float(f(v)))
    #yax = np.array(yax)
    
    
    # Finding peaks
   # peaks,_ = find_peaks(yax, peakHeight, prominence = 0.7)
    
    #if peak > len(peaks):
    #    return None
    hh1arr1=peak_interp(corr,ngrid,  minfq = 500, maxfq = 10)
    idxrep=peak
    #peak power larger than red noise peak power
    count=0.
    #peak power of red noise larger than observed peak power
    count11=0.
    bins11=[]
    bins=[]
   
    for i in range(numlc):
        y = shuffle(yy)
        ntau,params,decay_constant, parallel=inp_param(ntau, ngrid, f, minfq, maxfq)
        if( algorithm == 'wwz'):
            wwt_removedx = libwwz.wwt(timestamps=tt,
                                 magnitudes=y,
                                 time_divisions=ntau,
                                 freq_params=params,
                                 decay_constant=decay_constant,
                                 method='linear',
                                 parallel=parallel)
        corr1x=correlation_nd(np.rot90(wwt_removedx[2]),np.rot90(wwt_removedx[2]))
        #hhx=np.rot90(corr1x).T/corr1x.max()
        #hh1x=np.rot90(hhx.T)
        #hh1xarr=np.abs(hh1x).sum(1)/np.abs(hh1x).sum(1).max()
        hh1xarr=hh1arr1=peak_interp(corr1x, ngrid, minfq = 500, maxfq = 10)
        bins.append(hh1xarr[idxrep])
        if ((hh1arr1[idxrep]/hh1xarr[idxrep])>1.):
            count=count+1.
        else:
            count11=count11+1.  
            bins11.append(hh1xarr[idxrep])
            
    
    return bins, bins11, count/numlc, count11/numlc
