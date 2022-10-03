#!/usr/bin/env python
# coding: utf-8

# # Nonlinear Carbon Dynamics

# In[1]:


from Module_model import *
from Module_widget import *
from Module_plot import *

import warnings
warnings.filterwarnings('ignore')

display(Param_Panel)


# In[2]:


# tv,Tv,Cv,Ct = model(0,0,"rcp60co2eqv3",0.3916)

# plt.figure(figsize=(10, 5))
# plt.plot(tv, Ct)
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon concentration (ppm)',fontsize = 18);
# plt.grid(linestyle=':')
# plt.savefig("test.png")
Image("test.png")

