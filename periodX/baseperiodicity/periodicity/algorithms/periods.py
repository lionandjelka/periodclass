
import numpy as np
import pandas as pd
import json
from scipy.signal import chirp, find_peaks, peak_widths
from traitlets.traitlets import Integer
from scipy import interpolate, optimize
from scipy.stats.mstats import mquantiles
from sklearn.utils import shuffle
from periodicity.utils.correlation import correlation_nd
from periodicity.algorithms.wavelets import *





###get QSO which have u,g,r,i, light curves >=100 points

def get_qso(set11):
    global fs_gp 
    sett = []    
    for set1 in range(len(set11)):
        demo_lc = fs_gp.get_group(str(set11[set1]))
        d0 = demo_lc[demo_lc['filter'] == 1].sort_values(by=['mjd']).dropna()
        d1 = demo_lc[demo_lc['filter'] == 2].sort_values(by=['mjd']).dropna()
        d2 = demo_lc[demo_lc['filter'] == 3].sort_values(by=['mjd']).dropna()
        d3 = demo_lc[demo_lc['filter'] == 4].sort_values(by=['mjd']).dropna()
        d4 = demo_lc[demo_lc['filter'] == 5].sort_values(by=['mjd']).dropna()      
        if (len(d0) >= 100) and (len(d1) >=100) and (len(d2) > 100) and (len(d3) >=100):
            sett.append(str(set11[set1]))
    return sett
#def outliers(time,flux):
# Compute the Z-Score for each point in the light curve
#  z_scores = np.abs((flux - np.mean(flux)) / np.std(flux))

# Define a threshold for outlier detection
#  threshold = 3.0

# Find the indices of the non-outlier points. 
#  good_indices = np.where(z_scores <= threshold)[0]

# Create a new array with only the non-outlier points
#  clean_flux = flux[good_indices]
#  clean_time = time[good_indices]
#  if len(clean_flux)<0.1*len(flux):
#     clean_time=time
#     clean_flux=flux
 # clean_err=err[good_indices]
#  return clean_time, clean_flux#,clean_err

def outliers(time,flux):
# Compute the Z-Score for each point in the light curve
  z_scores = np.abs((flux - np.mean(flux)) / np.std(flux))

# Define a threshold for outlier detection
  threshold = 3.0

# Find the indices of the non-outlier points. 
  good_indices = np.where(z_scores <= threshold)[0]

# Create a new array with only the non-outlier points
  clean_flux = flux[good_indices]
  clean_time = time[good_indices]
  if len(clean_flux)<0.1*len(flux):
     clean_time=time
     clean_flux=flux
 #THIS PART IS ADDITIONAL AS WE FOUND THAT SOME POINTS COULD BE PROBLEMATIC EVEN AFTER 3SIGMA
 # clean_err=err[good_indices]
      # Find the differences between consecutive points
  differences = np.diff(clean_flux)

    # Find the index of the outlier point
  outlier_index = np.argmax(differences) + 1
  if np.max(differences)>0.3:
    # Remove the outlier point
   cleaned_mag = np.delete(clean_flux, outlier_index)
   cleaned_time=np.delete(clean_time, outlier_index)
  else:
   cleaned_time=clean_time
   cleaned_mag=clean_flux
  return cleaned_time, cleaned_mag#,clean_err

def get_lc22(set1):
    global fs_gp  # Declare fs_gp as a global variable
    fs_gp = shared_data_loader.fs_gp
    demo_lc = fs_gp.get_group(set1)
    d0 = demo_lc[(demo_lc['filter'] == 1) ].sort_values(by=['mjd']).dropna()
    d1 = demo_lc[(demo_lc['filter'] == 2) ].sort_values(by=['mjd']).dropna()
    d2 = demo_lc[(demo_lc['filter'] == 3) ].sort_values(by=['mjd']).dropna()
    d3 = demo_lc[(demo_lc['filter'] == 4) ].sort_values(by=['mjd']).dropna()
  #  d4 = demo_lc[(demo_lc['filter'] == 5) ].sort_values(by=['mjd'])
    tt00 = d0['mjd'].to_numpy()
    yy00 = d0['psMag'].to_numpy()
    tt11 = d1['mjd'].to_numpy()
    yy11 = d1['psMag'].to_numpy()
    tt22 = d2['mjd'].to_numpy()
    yy22 = d2['psMag'].to_numpy()
    tt33 = d3['mjd'].to_numpy()
    yy33 = d3['psMag'].to_numpy()
   # tt4 = d4['mjd'].to_numpy()
   # yy4 = d4['psMag'].to_numpy()
    tt0,yy0=outliers(tt00,yy00)
    print('ppp')
    tt1,yy1=outliers(tt11,yy11)
    print('p1')
    tt2,yy2=outliers(tt22,yy22)
    tt3,yy3=outliers(tt33,yy33)
    sampling0 = np.mean(np.diff(tt0))
    sampling1 = np.mean(np.diff(tt1))
    sampling2 = np.mean(np.diff(tt2))
    sampling3 = np.mean(np.diff(tt3))
  #  sampling4 = np.mean(np.diff(tt4))
    return tt0, yy0, tt1, yy1, tt2, yy2, tt3, yy3, sampling0, sampling1, sampling2, sampling3


def same_periods(r_periods0,r_periods1,up0,low0, up1,low1,peaks0,hh0,tt0,yy0, peaks1, hh1,tt1,yy1):
    r_periods0=np.array(r_periods0)
    r_periods1=np.array(r_periods1)
    up0=np.array(up0)
    low0=np.array(low0)
    up1=np.array(up1)
    low1=np.array(low1)
    number_of_lcs = 50  # Number of LCs used for a simulation
    print('prip')
    if  len(r_periods0)==len(r_periods1):
        index=np.where(np.isclose(r_periods0, r_periods1, rtol = 1e-01))
          #r_periods=r_periods0[np.isclose(r_periods0, r_periods1, rtol = 1e-01)]
        r_periods=np.take(r_periods0, index[0])
        #r_periods=r_periods0[index]
        up=list(np.take(up0,index[0]))
        low=list(np.take(low0,index[0]))
        # Determine significance
        sig=[]
        if len(r_periods.tolist())>0:
         for i in range(len(index[0])):
          peak_of_interests =index[0][i] # For wich peak we calculate significance
          bins, bins11, sig1, siger = signif_johnoson(number_of_lcs, peak_of_interests ,peaks0, hh0, tt0, yy0, 80,800)
          sig.append(1.-siger)
    elif len(r_periods0) < len(r_periods1):
        index_l= np.where(np.isclose(np.resize(r_periods0,len(r_periods1)), r_periods1, rtol = 1e-01)) 
        sig=[]          
       # r_periods = np.zeros(len(index_l[0] ))
        #up=np.zeros(len(index_l[0] ))
        #low=np.zeros(len(index_l[0] )) 
        r_periods = np.take(r_periods1, index_l[0])
         # up=up1[index_l[i]]
          #low=low1[index_l[i]]
        up=np.take(up1,index_l[0])
        low=np.take(low1, index_l[0])
        if len(r_periods.tolist())>0:
         for i in range(len(index_l[0])):
          peak_of_interests = index_l[0][i] # For wich peak we calculate significance
          bins, bins11, sig1, siger = signif_johnoson(number_of_lcs, peak_of_interests ,peaks1,hh1, tt1, yy1, 80,800)
          sig.append(1.-siger)

          print('x')  
    elif len(r_periods0) > len(r_periods1): 
        print(len(r_periods0), len(r_periods1))
        index_g= np.where(np.isclose(np.resize(r_periods1,len(r_periods0)), r_periods0, rtol = 1e-01)) 
        sig=[]          
       # r_periods = np.zeros(len(index_g[0] ))
       # up=np.zeros(len(index_g[0] ))
       # low=np.zeros(len(index_g[0])) 
        r_periods = np.take(r_periods0, index_g[0])
        up=np.take(up0,index_g[0])
        low=np.take(low0, index_g[0])
        if len(r_periods.tolist())>0:
         for i in range(len(index_g[0])):
         # up=up1[index_l[i]]
          #low=low1[index_l[i]]       
          peak_of_interests = index_g[0][i] # For wich peak we calculate significance
          bins, bins11, sig1, siger = signif_johnoson(number_of_lcs, peak_of_interests ,peaks0,hh0, tt0, yy0, 80,800)
          sig.append(1.-siger) 
        #r_periods=r_periods0[index_g]
        #up=up0[index_g]
        #low=low0[index_g]
        #sig=[]
        #if len(r_periods.tolist())>0:
         #for i in range(len(index)):
          #peak_of_interests = peaks0[index_g[i]] # For wich peak we calculate significance
          #bins, bins11, sig1, siger = signif_johnoson(number_of_lcs, peak_of_interests ,peaks0, hh0, tt0, yy0, 80,800)
          #sig.append(sig1)


    return np.array(r_periods), np.array(up),np.array(low), np.array(sig)
def process1(set1):
    global shared_data_loader
    fs_gp = shared_data_loader.fs_gp
    det_periods=[]
    tt0,yy0, tt1,yy1,tt2,yy2,tt3,yy3,sampling0,sampling1,sampling2,sampling3=get_lc22(set1)
    wwz_matrx0,  corr0, extent0 = hybrid2d(tt0, yy0, 80, 800, minfq=2000., maxfq=10.)
    peaks0,hh0,r_periods0, up0, low0 = periods (int(set1), corr0, 800, plot=False)
    wwzmatrx1, corr1, extent1 = hybrid2d(tt1,yy1, 80, 800, minfq=2000., maxfq=10.)
    peaks1,hh1,r_periods1, up1, low1 = periods (int(set1), corr1, 800, plot=False)
    wwz_matrx2,  corr2, extent2 = hybrid2d(tt2, yy2, 80, 800, minfq=2000., maxfq=10.)
    peaks2,hh2,r_periods2, up2, low2 = periods (int(set1), corr2, 800, plot=False)
    wwzmatrx3, corr3, extent3 = hybrid2d(tt3,yy3, 80, 800, minfq=2000., maxfq=10.)
    peaks3,hh3,r_periods3, up3, low3 = periods (int(set1), corr3, 800, plot=False)
    r_periods01, u01,low01,sig01=same_periods(r_periods0,r_periods1,up0,low0,up1,low1,peaks0,hh0,tt0,yy0, peaks1,hh1,tt1,yy1)
    print(set1)
   # for j in range(len(r_periods01)):
    if  r_periods01.size>0 and  u01.size>0 and low01.size>0 and sig01.size>0: 
      for j in range(len(r_periods01.ravel())):
        #u=1,g=2,r=3,i=4,z=5
        det_periods.append([int(set1),sampling0,sampling1,r_periods01[j], u01[j],low01[j], sig01[j], 12])
    elif r_periods01.size==0:
        det_periods.append([int(set1),0,0,0, 0,0,0,12])
   
    r_periods02, u02,low02,sig02=same_periods(r_periods0,r_periods2,up0,low0,up2,low2,peaks0, hh0,tt0,yy0, peaks2,hh2,tt2,yy2)

   # for j in range(len(r_periods01)):
    if  r_periods02.size>0 and u02.size>0 and low02.size>0 and sig02.size>0:
      #ur=02
      for j in range(len(r_periods02.ravel())):
       det_periods.append([int(set1),sampling0,sampling2,r_periods02[j], u02[j],low02[j],sig02[j], 13])
    elif r_periods02.size==0:
       #ur=02
       det_periods.append([int(set1),0,0,0, 0,0,0,13])
    r_periods03, u03,low03,sig03=same_periods(r_periods0,r_periods3, up0,low0, up3,low3,peaks0, hh0,tt0,yy0, peaks3,hh3,tt3,yy3)
    if  r_periods03.size  and  u03.size>0 and low03.size>0 and sig03.size>0:
     for j in range(len(r_periods03.ravel())):
       #ui=03
      det_periods.append([int(set1),sampling0,sampling3,r_periods03[j], u03[j],low03[j],sig03[j],14]) 
    elif r_periods03.size==0:
      #ui=03
      det_periods.append([int(set1),0,0,0, 0,0,0,14])
    return np.array(det_periods)

def get_full_width(x: np.ndarray, y: np.ndarray, peak:np.ndarray,height: float = 0.5) -> float:
    """Function is used to calculate the error of the determined period using FWHM method.
        The period uncertainty method of Schwarzenberg-Czerny requires the so called Mean Noise Power Level (MNPL)
        in the vicinity of P. The 1-sigma confidence interval on P then is equal to the width of the line at the P – MNPL level. 
        This method is a so-called ‘post-mortem analysis’. 
        To find the MNPL we detect FWHM of the peak and then use mquantile method to detect points between 25th and 75th quantile 
        Reference: Schwarzenberg-Czerny, A., 1991, Mon. Not. R. astr. Soc., 253, 198-206
        
        Parameters
        ----------
        x,y: numpy array, arrays with data
        peak : numpy array, array with determined peaks
        height: size of peak
        
        Returns:
        ----------
        Arrays with results, and quantiles and half peak size, and low/high x values

    """

    er1=[]
    er3=[]
    height_half_max = 0
    quantiles = []
    phmax =  []
    x_lows =[]
    x_highs = []
    # Iterate over all peaks
    for i in range(len(peak)):
        
        # Half max of a peak as its y axis value multiplied by height parameter (by default 0.5)
        height_half_max=y[peak[i]]*height
        index_max = peak[i]
        
        # Determine the x-axis border of a peak but on its half max height    
        tmp = peak[i]
        tmp2 = peak[i]
                                          
        x_low = 0
        x_high = 0
        while True:
            tmp = tmp - 1;
            if( y[tmp] - height_half_max) < 0:
                x_low = x[tmp+1]
                break
        
        while True:
            tmp = tmp + 1;
            if( y[tmp] - height_half_max) < 0:
                x_high = x[tmp-1]
                break
        
            
        # q1 and q2 represents quantiles    
        q25 = 0
        q75 = 0
        
        # Check if we have sufficient number of points (5 points)
        
        if( index_max - 5 > 0 ):

            arr=y[(x>=x_low)&(x<=x_high)]  # array of data between x_low and x_hight where we determine quantiles
            # For a reference see  Schwarzenberg-Czerny, A., 1991, Mon. Not. R. astr. Soc., 253, 198-206
            
            q25,q75 = mquantiles(arr, [0.25,0.75])
 
            # Calculate the value at specific quantiles (q1 and q3)
            inversefunction = interpolate.interp1d(y[index_max-5:index_max],  x[index_max-5:index_max], kind='cubic',fill_value="extrapolate")
            inversefunction2 = interpolate.interp1d(y[index_max:index_max+5],  x[index_max:index_max+5], kind='cubic',fill_value="extrapolate")
            
            xer1=inversefunction(q25)
            xer3=inversefunction2(q75)
            
        else:
            xer1 = 0
            xer3 = 0
            
        er1.append(xer1)
        er3.append(xer3)
        quantiles.append([q25, q75])
        phmax.append(height_half_max)
        x_lows.append(x_low)
        x_highs.append(x_high)
        
        
    return er1,er3, quantiles, phmax, x_lows, x_highs



def periods (lcID, data, ngrid, plot = False, save = False, peakHeight = 0.6, prominence = 0.7, minfq = 500, maxfq = 10, xlim = None): 
    """Perform period determination for the output of hybrid2d data.

        Parameters
        ----------
        lcId : id of a curve
        data :auto correlation matrix
        ngrid : values for controling wwz execution (see inp_param function)
        minfq : minimum frequency
        maxfq : maximum fraquency
        peakHeight: max peak height
        prominence: prominence for peak determination
        plot: True of Folse if we want a plot
        save: determine to save a plot
        xlim: set xlim for a plot
        
        Returns:
        ---------
        r_peaks: arrays with determined periods
        r_peaks_err_upper: arrays with upper errors of corresponding periods
        r_peaks_err_lower: arrays with lower errors of corresponding periods

    """
    
    hh1=np.rot90(data).T/np.rot90(data).T.max()
    hh1arr=np.rot90(hh1.T)
    hh1arr1=np.abs(hh1arr).sum(1)/np.abs(hh1arr).sum(1).max()
    
    
    
    fmin =1/minfq
    fmax = 1/maxfq
    df = (fmax - fmin) / ngrid
    
    
    
    # osax interpolation (stacked data of h2d along one axis) to obtain more points
    osax=np.arange(start=fmin,stop=fmax+df,step=df)
    xax =np.arange(start=fmin, stop = fmax + df, step = df/2)
    from scipy import interpolate
    f = interpolate.interp1d(osax, np.abs(hh1arr1), fill_value="extrapolate")
    yax = []
    for v in xax:
        yax.append(float(f(v)))
    yax = np.array(yax)
    
    
    # Finding peaks
    peaks,_ = find_peaks(yax, peakHeight, prominence = prominence)

    
    # Polotting if needed
    if( plot == True ):       
        if xlim != None:
            plt.xlim(xlim)
        plt.plot(xax,np.abs(yax))
        plt.axvline(xax[peaks[0]],ymin=0,ymax=1, linestyle='--', color='k')
        plt.title(str(lcID))
        plt.xlabel(r'Freqeuncy [day$^{-1}$]')
        plt.ylabel(r"correlation")
        if( save == True) :
            plt.savefig(str(lcID) + 'stackd_h2d.png')  
        
        
    
    #Get error estimates for each peak (period)
    error_upper, error_lower, quantiles, halfmax, x_lows, x_highs = get_full_width (xax,yax,peaks)
    
    if plot == True:
        plt.plot(xax,np.abs(yax))
        if xlim != None:
            plt.xlim(xlim)
        plt.title(str(lcID))
        plt.xlabel(r'Freqeuncy [day$^{-1}$]')
        plt.ylabel(r"correlation")
        
        for i in range(len(peaks)):
            plt.axvline(xax[peaks[i]],ymin=0,ymax=1, linestyle='--', color='black')
            plt.axhline(quantiles[i][0],linestyle='--', color='green')
            plt.axhline(quantiles[i][1],linestyle='--', color='red')
            plt.axvline(x_lows[i], ymin =0, ymax = 1, linestyle ='--', color='blue')
            plt.axvline(x_highs[i], ymin =0, ymax = 1, linestyle ='--', color='blue')
            plt.axhline(halfmax[i],linestyle='--', color='purple')
        
        
        if( save == True) :
            plt.savefig(str(lcID) + 'stackd_h2d_peaks.png')  
        
        
    
    
    # Prepare the output
    r_peaks = []
    r_peaks_err_upper = []
    r_peaks_err_lower = []
    idx_peaks=[]
    for i in range(len(peaks)):
        r_peaks.append(1/xax[peaks[i]])
        idx_peaks.append(peaks[i])
        if( error_upper[i] == 0):
            r_peaks_err_upper.append(-1)
        else:
            r_peaks_err_upper.append(np.abs(1/xax[peaks[i]]-(1/error_upper[i])))       
        if(error_lower[i] == 0):
            r_peaks_err_lower.append(-1)
        else:
            r_peaks_err_lower.append(np.abs(1/xax[peaks[i]]-( 1/error_lower[i])))
    
    
        
    return idx_peaks, yax, r_peaks, r_peaks_err_upper, r_peaks_err_lower
    
    
    
def signif_johnoson(numlc, peak, idx_peaks, yax, tt, yy, ntau,ngrid, f = 2, peakHeight = 0.6, minfq = 500, maxfq = 10, algorithm ='wwz', method = 'linear'):
    """Determination of significance usign Johnson method

        Parameters
        ----------
        numlc : int, number of lc for determination
        peak : determined periodicity peak
        ### removed corr : hybrid 2d output
        peaks: array of indeces of detected peaks from period procedure
        hh1arr1: imported periodogram like structure from periods
        tt : time
        yy: magnitude
        plot: True of Folse
        ntau, ngrid, f : values for controling wwz execution (see inp_param function)
        minfq : minimum frequency
        maxfq : maximum fraquency
        peakHeight: max peak height,
        algorithm: wwz or superlets (for now)
  
    """
  #  hh1=np.rot90(corr).T/np.rot90(corr).T.max()
  #  hh1arr=np.rot90(hh1.T)
  #  hh1arr1=np.abs(hh1arr).sum(1)/np.abs(hh1arr).sum(1).max()
  #  peaks,_ = find_peaks(hh1arr1,peakHeight, prominence = 0.6)
    
   # if peak > len(peaks):
   #     return None
    
    idxrep=idx_peaks[peak]
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
        hhx=np.rot90(corr1x).T/corr1x.max()
        hh1x=np.rot90(hhx.T)
        hh1xarr=np.abs(hh1x).sum(1)/np.abs(hh1x).sum(1).max()

        bins.append(yax[idxrep])
        if ((yax[idxrep]/hh1xarr[idxrep])>1.):
            count=count+1.
        else:
            count11=count11+1.  
            bins11.append(hh1xarr[idxrep])
            
    
    return bins, bins11, count/numlc, count11/numlc
