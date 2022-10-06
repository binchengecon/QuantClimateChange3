#!/usr/bin/env python
# coding: utf-8

# # Nonlinear Carbon Dynamics

# In[1]:


from PIL import Image
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

@interact
def image(pulse=(0, 99), cearth=np.array((0.3725, 0.3916, 15.0)), baseline=["rcp60co2eqv3.csv", "rcp00co2eqv3.csv", "carbonvoid.csv"], year=np.array((1801, 2010)),option=["Trad","IRF","IRFstand","Tera"]):
    Figure_Dir = "./figure/NC_PulseExp/"
    # baseline = "rcp60co2eqv3.csv"
    # cearth=15.0
    # year=1801

    if option=="Trad":
        filename = Figure_Dir+"Baseline="+baseline+",cearth=" +             str(cearth)+",year="+str(year)+",pulse="+str(pulse)+".png"
    if option == "IRF":
        filename = Figure_Dir+"Baseline="+baseline+",cearth=" +             str(cearth)+",year="+str(year)+",pulse=" +             str(pulse)+",2IRF.png"    # return Image.open(filename)
    if option =="IRFstand":
        filename = Figure_Dir+"Baseline="+baseline+",cearth=" +             str(cearth)+",year="+str(year)+",pulse=" +             str(pulse)+",2IRF,per.png"    # return Image.open(filename)
    if option =="IRFTera":
        filename = Figure_Dir+"Baseline="+baseline+",cearth=" +             str(cearth)+",year="+str(year)+",pulse=" +             str(pulse)+",2IRF,Tera.png"    # return Image.open(filename)
    userimage = Image.open(filename)
    userimage_resize = userimage.resize((800, 1000))
    return userimage_resize

