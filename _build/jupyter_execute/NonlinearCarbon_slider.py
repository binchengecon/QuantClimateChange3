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
def image(pulse=(0, 99), cearth=np.array((0.3725, 0.3916, 15.0)), baseline=["rcp60co2eqv3.csv", "rcp00co2eqv3.csv", "carbonvoid.csv"], year=np.array((1801, 2010))):
    Figure_Dir = "./figure/NC_PulseExp/"
    # baseline = "rcp60co2eqv3.csv"
    # cearth=15.0
    # year=1801
    filename = Figure_Dir+"Baseline="+baseline+",cearth=" +         str(cearth)+",year="+str(year)+",pulse="+str(pulse)+".png"
    return Image.open(filename)


# In[3]:


pulse=1
Figure_Dir = "./figure/NC_PulseExp/"
# Figure_Dir = "./NC_PulseExp/"
# Figure_Dir = ""
cearth=15.0
baseline = "rcp60co2eqv3.csv"
year = 1801

filename = Figure_Dir+"Baseline="+baseline+",cearth=" +     str(cearth)+",year"+str(year)+",pulse="+str(pulse)+".png"
# Image("test.png")
filename
# Image(filename)
Image("./figure/NC_PulseExp/Baseline=rcp60co2eqv3.csv,cearth=0.3725,year1801,pulse=29.png")


# In[6]:


from PIL import Image
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
get_ipython().run_line_magic('matplotlib', 'inline')

img = Image.open('test.png').convert('L')


@interact
def binarize(th: (0, 255, 1)):
    return img.point(lambda p: 255 if p > th else 0)

