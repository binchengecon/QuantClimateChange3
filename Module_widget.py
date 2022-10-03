#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the code for the Jupyter widgets. It is not required
for the model framework. The widgets are purely for decorative purposes.
"""

#######################################################
#                    Dependencies                     #
#######################################################


from ipywidgets import widgets, Layout, Button, HBox, VBox, interactive
from IPython.core.display import display
from IPython.display import clear_output, Markdown, Latex
from IPython.display import Javascript
import numpy as np
try:
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot


# Define global parameters for parameter checks
params_pass = False
model_solved = False


#######################################################
#          Jupyter widgets for user inputs            #
#######################################################

## This section creates the widgets that will be diplayed and used by the user
## to input parameter values.

style_mini = {'description_width': '5px'}
style_short = {'description_width': '100px'}
style_med = {'description_width': '180px'}
style_long = {'description_width': '200px'}

layout_mini =Layout(width='18.75%')
layout_50 =Layout(width='50%')
layout_med =Layout(width='70%')

widget_layout = Layout(width = '100%')

pulse_size = widgets.BoundedFloatText( ## risk free rate
    value=10,
    min = 0,
    max = 100,
    step=0.5,
    disabled=False,
    description = 'Size of Pulse',
    style = style_med,
    layout = layout_med
)


pulse_year = widgets.BoundedFloatText( ## risk free rate
    value=1801,
    min = 1800,
    max = 2800,
    step=10,
    disabled=False,
    description = 'Year of Pulse',
    style = style_med,
    layout = layout_med
)


pulse_baseline = widgets.Dropdown(
    options = {"rcp00co2eqv3", "rcp60co2eqv3"},
    # value = 1,
    description='Carbon Concentration Baseline:',
    disabled=False,
    style = style_med,
    layout = layout_med
)


cearth = widgets.BoundedFloatText( ## risk free rate
    value=0.3916,
    min = 0.33,
    max = 20,
    step=0.01,
    disabled=False,
    description = 'Heat Capacity',
    style = style_med,
    layout = layout_med
)

runSim = widgets.Button(
    description='Run simulation',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

# runSlider = widgets.Button(
#     description='Run models',
#     disabled=False,
#     button_style='', # 'success', 'info', 'warning', 'danger' or ''
# )

box_layout       = Layout(width='100%', flex_flow = 'row')#, justify_content='space-between')
box_layout_wide  = Layout(width='100%', justify_content='space-between')
box_layout_small = Layout(width='10%')

Box_pulse = VBox([widgets.Label(value="Pulse"), pulse_size, pulse_year,pulse_baseline,cearth], layout = Layout(width='90%'))


line1      = HBox([Box_pulse], layout = box_layout)
Param_Panel = VBox([line1])

sim_var_names_external = ['Temperature Anomaly', 'Carbon Concentration Dynamics', 'Carbon Emission', 'Impulse Response Function']

simulate_external = widgets.SelectMultiple(options = sim_var_names_external,
    value = ['Temperature Anomaly'],
    rows = len(sim_var_names_external),
    disabled = False
)

simulate_box_external = VBox([widgets.Label(value="Select variables to simulate:"),simulate_external], layout = Layout(width='100%'))

run_box_sim = VBox([widgets.Label(value="Run simulation"), runSim], layout = Layout(width='100%'))

simulate_box_external_run = HBox([simulate_box_external, run_box_sim], layout = Layout(width='100%'))

