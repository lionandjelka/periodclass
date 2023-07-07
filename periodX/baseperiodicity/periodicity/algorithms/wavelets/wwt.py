import libwwz
import numpy as np
from periodicity.utils.correlation import correlation_nd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys
import io
from traitlets.traitlets import Integer






def compute_frequency_grid(Nn, minfq = 500, maxfq = 10):
    fmin =1/minfq
    fmax = 1/maxfq
    df = (fmax - fmin) / Nn
    return df, fmin,fmax



def inp_param(ntau,ngrid, f = 2, minfq = 500, maxfq = 10):
   
    """Calculate the imput parameteres of WWZ

        Parameters
        ----------
        ntau : int, number of time
        ngrid: grid size
        f: for calculation of decay constant
        minfq:minimum frequnecy default 1/500 days
        maxfq: maximum frequnecy default 1/10days
    """
    
    df, fmin, fmax = compute_frequency_grid(ngrid, minfq, maxfq)
    
    frequency_low = fmin
    frequency_high = fmax
    frequency_steps = df

    # Set the override to False (Setting to True will ignore the low and high frequency limitations)
    override = False

    # gather the frequency parameters into a list [freq_low, freq_high, freq_step, override]
    frequency_parameters = [frequency_low, frequency_high, frequency_steps, override]

    # We will then select the decay constant for our analyzing wavelet (should be < 0.2), where c = 1/(2*w^2) 
    # The analyzing wavelet decays significantly in a single cycle 2*pi/w, where w = 2*pi*f
     #f = 3.# we choose 3-4 since our signal of interest

    w = 2 * np.pi * f
    decay_constant = 1/(2*w**2)

    # Finally, we select to wether to run with parallization (recommend True)
    parallel = True
    return ntau,frequency_parameters,decay_constant, parallel




def wwt(tt, mag,ntau,ngrid, f = 2, minfq = 500, maxfq = 10,  method = 'linear'):
    """Calculate the wwz of the given signal

        Parameters
        ----------
        tt : list of time data
        mag : list of magnitude values
        ntau, ngrid : values for controling wwz execution (see inp_param function)
        minfq : minimum frequency 1/minfreq
        maxfq : maximum fraquency 1/maxfq
        method : "linear" / "octave"

        """
    
    ntau,params,decay_constant, parallel=inp_param(ntau, ngrid, f, minfq, maxfq)
    
    return libwwz.wwt(timestamps=tt, magnitudes=mag,
                                 time_divisions=ntau,
                                 freq_params=params,
                                 decay_constant=decay_constant,
                                 method=method,
                                 parallel=parallel)
    



def hybrid2d(tt, mag, ntau,ngrid, f = 2, minfq = 500, maxfq = 10,  method = 'linear'):
    """Perform hybrid2d method on given data 

        Parameters
        ----------
        tt : list of time data
        mag : list of magnitude values
        ntau, ngrid : values for controling wwz execution (see inp_param function)
        minfq : minimum frequency
        maxfq : maximum fraquency
        method : "linear" / "octave"

    """
    
    #Perform wwz of data
    wwz_matrix = wwt(tt, mag, ntau, ngrid, f, minfq, maxfq, method)
    
    #autocrrelate matrix
    corr = correlation_nd(np.rot90(wwz_matrix[2]),np.rot90(wwz_matrix[2]))
    extentmin=np.min(wwz_matrix[1])
    extentmax=np.max(wwz_matrix[1])

    extent=[extentmin,extentmax,extentmin,extentmax]
    
    

    return wwz_matrix, corr,  extent



# OLF FUNCTIONS TO BE DELETED

# Decay constant default value is taken from Foster, 1996



# def error_determination (x: np.ndarray, y: np.ndarray, peak:np.ndarray,height: float = 0.5) -> float:
#     height_half_max=y[peak[0]]*height
#     index_max = peak[0]
    

    
#     x_low = np.interp(height_half_max, y[0:index_max],  x[0:index_max])
#     x_high = np.interp(height_half_max, np.flip(y[index_max:index_max+5]), np.flip(x[index_max:index_max+5]))
    
    
#     arr=y[(x>=x_low)&(x<=x_high)]
    
#     q3,q1 = np.quantile(arr, [0.86,0.14])
    
#     er1 = np.interp(q1, y[:index_max], x[:index_max])
#     er3 = np.interp(q3, y[:index_max], x[:index_max])
    
#     return 1./x_low,1./x_high



# def wwt_parameters(t, n):
#     pmx=(t.max()-t.min())/2.
#     pmn=np.min(np.diff(t))
#     fmin = pmn
#     fmax = pmx
#     df = (fmax - fmin) / 500
    
#     params = [fmin, fmax, df, False]
#     return params

# from scipy.signal import find_peaks


# def wwt_find_statistical_significance(tt, yy, ndat, hh1arr1, params, nd = 1000, peaks= 0):
    
#     mccount=nd
#     idxrep=peaks

#     count=0.
#     count11=0.
#     bins11=[]
#     bins=[]
#     for i in range(mccount):
#         y = shuffle( yy)
#         wwt_removedx, _ = wwt(tt,y,ndat, params)
#         corr1x=correlation_nd(np.rot90(wwt_removedx[2]),np.rot90(wwt_removedx[2]))
#         hhx=np.rot90(corr1x).T/corr1x.max()
#         hh1x=np.rot90(hhx.T)
#         hh1xarr=np.abs(hh1x).sum(1)/np.abs(hh1x).sum(1).max()
#         bins.append(hh1xarr[idxrep])
#         if ((hh1arr1[idxrep]/hh1xarr[idxrep])>1.):
#             count=count+1.
#         else:
#             count11=count11+1.  
#             bins11.append(hh1xarr[idxrep])
    
#     return (count/mccount, count11/mccount)   

# def wwt_maxpeak(hh1arr1, prominence = 0.995):
#     step = 0.005
#     while True:
#         peaks, _ = find_peaks(hh1arr1, prominence=prominence)
#         if (len(peaks) == 0):
#             prominence = prominence - step
#         else:
#             break
    
#     return peaks
        


# def wwt_find_freq(wwt_res, n): 
#     corr = correlation_nd(np.rot90(wwt_res[2]),np.rot90(wwt_res[2]))
#     extentmin=np.min(wwt_res[1])
#     extentmax=np.max(wwt_res[1])
#     extent=[extentmin,extentmax,extentmin,extentmax]
    
#     hh1=np.rot90(corr).T/corr.max()
#     hh1arr=np.rot90(hh1.T)
    
#     hh1arr1=np.abs(hh1arr).sum(1)/np.abs(hh1arr).sum(1).max()
    
#     n = len(np.abs(hh1arr).sum(1))
#     osax=np.linspace(extent[0],extent[1],n)
#     peaks = wwt_maxpeak(hh1arr1)
    
    
#     plt.plot(osax,np.abs(hh1arr).sum(1))
#     plt.axvline(osax[peaks[0]],ymin=0,ymax=1)
#     return peaks, 1./osax[peaks[0]], osax, hh1arr1 ;
    
    
    


# # def wwt(tt, mag, n, params,  method = 'linear', parallel = True, output = True):
    
# #     save_stdout = sys.stdout
# #     if not output:
# #         sys.stdout = open('trash', 'w')
# #     f=4
# #     w = 2 * np.pi * f
# #     decay_constant = 1/(2*w**2)
# # #     params = [fmin, fmax, df, True] # wwt_parameters(tt,n)
# #     res = libwwz.wwt(timestamps=tt, magnitudes=mag,
# #                                  time_divisions=n,
# #                                  freq_params=params,
# #                                  decay_constant=decay_constant,
# #                                  method=method,
# #                                  parallel=parallel)
# #     if not output:
# #         sys.stdout = save_stdout
        
# #     return res, wwt_parameters(tt,n)
