#!/usr/bin/env python
# coding: utf-8

# # Nonlinear Carbon Dynamics

# In[1]:


from Module_widget import *
from Module_model import *
from Module_plot import *

import warnings
warnings.filterwarnings('ignore')

display(Param_Panel)


# In[2]:


pulsesize = pulse_size.value
pulseyear = pulse_year.value
pulsebaseline=pulse_baseline.value
pulsecearth = cearth.value



display(simulate_box_external_run)


# In[3]:


all_var_names = ['Temperature Anomaly', 'Carbon Concentration Dynamics', 'Carbon Emission', 'Impulse Response Function']
selected_index = [all_var_names.index(element) for element in simulate_external.value]
# print(selected_index)

# for j in selected_index:
#     print(j+1)

fig, ax = plot_simulation(pulsesize, pulseyear,pulsebaseline,pulsecearth, selected_index)
plt.tight_layout()
plt.show()

