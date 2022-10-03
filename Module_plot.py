import numpy as np
np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt
from Module_model import model
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from plotly.offline import init_notebook_mode, iplot
from IPython.display import Image

    

def plot_simulation(pulse, year,baseline,cearth, selected_index):
    colors = ['blue', 'green', 'red', 'gold', 'cyan', 'magenta', 'yellow', 'salmon', 'grey', 'black']
    titles = ['Temperature Anomaly T', 'Carbon Concentration Dynamics C', 'Carbon Emission G', 'Impulse Response Function']
    ylabels = ['Temperature (K)', 'Carbon (ppm)', 'Emission (Gtc)', 'Degree (Celsius)']
    fig, axs = plt.subplots(len(selected_index),1, figsize = (3*(len(selected_index)),20), dpi = 200)

    modelsol= model(pulse, year,baseline,cearth)

    for j in selected_index:

        axs[j].plot(modelsol[0], modelsol[j+1],color=colors[j], label=f"cearth={cearth},pulse={pulse}")
        axs[j].set_ylabel(ylabels[j])
        axs[j].set_title(titles[j])
        axs[j].set_xlabel('Year')
        axs[j].legend(loc = 'lower right')

    return fig, axs
    
