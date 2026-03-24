import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# We store the slider in a global list or dictionary to prevent garbage collection
persistent_widgets = []

def plot_3d_stack(data_stack):
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.2) # Make room for slider
    
    idx = 0
    im = ax.imshow(data_stack[idx], cmap='viridis', vmin=data_stack.min(), vmax=data_stack.max())
    ax.set_title(f"Slice: {idx}")

    # Create slider axes
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    # Assign to a variable
    slider = Slider(
        ax=ax_slider,
        label='Index ',
        valmin=0,
        valmax=data_stack.shape[0] - 1,
        valinit=idx,
        valfmt='%0.0f'
    )

    def update(val):
        curr_idx = int(slider.val)
        im.set_data(data_stack[curr_idx])
        ax.set_title(f"Slice: {curr_idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    # CRITICAL: Keep the reference alive
    persistent_widgets.append(slider)
    
    plt.show()