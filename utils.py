"""
Utility Functions for Data Visualization

Team Zephyrus - OlympAI Hackathon 2025
Bright Riders School, Abu Dhabi

This provides helper functions for converting matplotlib figures
into base64 encoded strings that can be embedded in HTML/JSON responses.

Functions:
    create_plot_base64: Convert matplotlib figure to base64 string
"""

import io
import base64
import matplotlib.pyplot as plt


def create_plot_base64(fig: plt.Figure) -> str:
    
    # create buffer in memory
    img_buffer = io.BytesIO()
    
    # save figure to buffer as PNG
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    
    # go back to start of buffer
    img_buffer.seek(0)
    
    # convert to base64 and return as string
    encoded = base64.b64encode(img_buffer.getvalue()).decode()
    
    return encoded
