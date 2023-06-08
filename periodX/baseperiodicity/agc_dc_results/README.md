<ul>
<li> 
 
<b> first_run_22_09_2022.csv</b>: Preliminary results performed on AGN Data Challenge data. From the results it can be seen that the determination of the period errors can be strongly influenced by the peak width. Therefore, we need to apply some checking of the peak width and then we have to discard one error. For more clarification please see the example given in the <a href="https://github.com/LSST-sersag/periodicities/blob/main/agc_dc_results/AGN_DC_example.ipynb"> notebook </a>.
</li>
</ul>
Here are the latest results (end of May 23/beginning of June 23).
<li>
 <b> all.csv </b>: Here are the latest results of the pipeline tested on BURA IDAC Croatia and AI Platforma Serbia for detected periods in given pairs of bands.
 </li>
 </ul>
 <li>
<b> iou.csv </b>: We accompany all.csv with metric Intersection over Union. For given band pairs we calculate IoU metric. As metric is closer to 1 it means that we get more overlapping of detcted periods and their relative error. 
 </li>
  </ul>
  Google colaboratory notebook with visulaization of results is available  <a href="https://colab.research.google.com/drive/1I7sl6W5x8yi8vR6c6QeuQPcvL3Yi3skV?usp=sharing"> here </a>.
