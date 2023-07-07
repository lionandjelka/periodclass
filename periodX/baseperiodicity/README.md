<b> TIME DOMAIN PERIODICITY MINING PIPELINE </b>
 Framework for 2D Hybrid Method; Q3 FY22 work done: Apr-Jun 2022,
preliminary repository https://github.com/LSST-sersag/periodicities


<ol> 
    <li> <a href="#intro"> Introduction to package. </a></li>
    <li> <a href="#inkind"> Introduction to in-kind proposal. </a></li>
    <li> <a href="#links"> Quick links </a> </li>
    
 </ol>
 
 
 <h3 id="intro"> Introduction to <u> periodicity </u> package </h3>
 
  <p>
    The package will be separated into a few different modules: </p>
    <ol>
        <li> read module. Read module is primarily used for data acquisition and pre-processing. It will fetch data (either from RSP or user data), and it will check if all necessary pieces of information are given, normalize it and create input objects that will be used as input in our pipeline. For the pipeline to function correctly, initial data must have Object ID, time, flux, flux uncertainty, multi-band LSST Lomb-Scargle periodicity, and it must be supplied by the user (or the connecting system, such as RSP). Read module will have three sub-modules. The first one, named lsst, will be intended for reading and obtaining data from lsst RSP. It will have different functions that will be able to reach lsst data remotely (using API or tas protocol) or directly in RSP. The second one, named user, will have similar functions but for the data supplied by the user. The third one is used for data normalization and basic pre-processing.  </li>
 
<li> Utils module. The purpose of this module is to store any functions that can be used as utilities, e.g., functions to simulate artificial light curves or any other calculation-based functions such as autocorrelation functions, etc. </li>

<li> Plots module. This module's primary purpose is for the graphical representation of data. It has functions to visualize input data, as well as output data. </li>

<li> Output module. This module's purpose is to obtain output data as Python objects. It will have functions that implement our pipeline to obtain periodic features of the LC. Also, it will have the ability to simultaneously execute a vast number of different LC to obtain catalogs with periodicities. To logically organize the module, it will have two sub-modules, one for obtaining periodicity features for a single LC and the other with functions for multiple light curves. </li>
        </ol>

<b> To start we recommend you to see our notebook toutorials: </b> <br>
<a href=""> Tutorial 1 </a>
<a href=""> Tutorial 2 </a>

<b> Also we recomend to see our results obtained using AGN DC data : <a href=""> link </a> </b>
  
  
 <h3 id="inkind">  Introduction to in-kind proposal.</h3>
 
 
<ul>
<li> 
The pipeline starts reading all the relevant LC information required by the periodicity study: Object ID, time, flux, flux
uncertainty, multiband LSST Lomb Scargle periodicities. As the starting point of the analysis, the LCs are checked in
relation to the number of points in LCs (our metrics show that LC with cadences <100obs/10yr should be discarded).
We are now working on the development of a method for uploading a large number of LCs utilizing PyTorch and
GPU.
</li>
<li> <b> Data processing </b>
The complex time series have varying scales, trends and outliers. In the first step, we perform data preprocessing
such as data normalization, detrending, and outlier processing. The time series detrending is a key step as the trend
component would bias the estimation of period, resulting in misleading information.
Detrended light curve (y) is further processed to coarsely remove extreme outliers by function Ψ( (𝑦𝑡 −𝜇 )/s), where 𝜇
and 𝑠 are the median and mean absolute deviation (MAD), and function is defined by Ψ(𝑥) =𝑠𝑖𝑔𝑛(𝑥)𝑚𝑖𝑛(|𝑥|,𝑐) with
tuning parameter c.
</li>
<li> <b>LC MODELING UNIT</b> 
Following sorting in 1), LC with cadences of <<1500obs/10yr are exposed to nonparametric modeling (e.g. neural
processes Hajdinjak et al 2021) and parametric modeling (Gaussian Processes, Kovacevic et al. 2017, 2018). The
modeling unit have been presented at AGN SC Summer meeting 2021 in talk and paper by Hajdinjak et al 2021.

</li>
<li><b> PERIODICITY MINING METHODS: </b>
Assumes time domain analysis: The LCs are passed into the following analysis stage which includes wavelet based
algorithms: continuous wavelet transform (CWT), the weighted wavelet Z-transform (WWZ), 2D Hybrid
technique, Bayesian Lomb Scargle Periodogram.
-Continuous wavelet transform (CWT, Torrence and Compo 1998). In the CWT analysis, we use the Morlet
mother function. We use the implementation provided in python which also provides the significance of
the peaks (e.g. Ackermann et al. 2015).
-The weighted wavelet Z-transform (WWZ, Foster 1996; Gupta et al. 2018). To calculate the significance,
we use the simulated light curves technique. To obtain the significance, LCs are simulated, based on the
best-fitting result of power spectral density and the probability density function of the original LC (see
Emmanoulopoulos et al, 2013). For each simulated LC, a WWZ power spectrum is calculated and
significance is determined as in Kovacevic et al 2020 .
For significance determination it is critical to determine how many light curves can be simulated, which may be
several thousands for wavelet-based methods due to processing capacity.
-2D Hybrid technique (Kovacevic et al 2018, 2019, Kovacevic et al 2020a, 2021}. is based on the calculation of the
Continuous wavelet transform (CWT) of light curves (LC). These CWTs are cross-correlated
𝑀 = 𝐶𝑜𝑟𝑟(𝐶𝑊𝑇(𝐿𝐶1), 𝐶𝑊𝑇(𝐿𝐶2)) and presented as 2D heatmaps (M). It is also possible to correlate CWT of one
light curve with itself and obtain a detection of oscillations in one light curve. M provides information about the
presence of coordinated or independent signals and relative directions of signal variations. To obtain the
significance, LCs are simulated, based on the best-fitting result of power spectral density and the
probability density function of the original LC (see Emmanoulopoulos et al, 2013). For each simulated LC,
2DHybrid maps are calculated and significance is determined as in Kovacevic et al 2020 .

</li>

<li> <b> STATISTICAL DECISION ON PERIODICITY MINING OUTPUT OF THE PIPELINE</b> </li> Finally, we can classify sources by imposing a following condition:
i) to choose Lower-Significance Candidate, at least three techniques must yield a detection at level L3 at the
same period
ii) we can impose a constraint on the highly significant periodicity candidates: four techniques must yield a
detection at level L4 at the same period
<li> <b> PRODUCTS </b> </li> 1) nonparametric and parametric modeled LC,
2) extracted periodic features of LC (periodicities, uncertainties, significance (probability) of periods, 2DHybrid
maps, BGLS maps)
Due to non parametric modeling of LC, the proposed pipeline is applicable to other objects LC, such as variable stars.

</ul>

![image](https://user-images.githubusercontent.com/78701856/191952700-d104bc04-72a4-4258-961b-2c139619e673.png)


 
 
 
 