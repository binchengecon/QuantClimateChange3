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
display(simulate_box_external_run)


# In[2]:


all_var_names = ['Temperature Anomaly', 'Carbon Concentration Dynamics', 'Carbon Emission', 'Impulse Response Function']
selected_index = [all_var_names.index(element) for element in simulate_external.value]
# print(selected_index)

# for j in selected_index:
#     print(j+1)

fig, ax = plot_simulation(pulse_size.value, pulse_year.value,pulse_baseline.value,cearth.value, selected_index)
plt.tight_layout()
plt.show()

