import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as hc
import numpy as np


def plot_linkage_data(linkage_matrix, color_threshold=None):
    plt.figure()
    plt.subplot(2, 1, 1)
    hc.dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        orientation='left',
        p=30,
        show_leaf_counts=True,
        leaf_rotation=0,
        leaf_font_size=8,
        show_contracted=True,
        color_threshold=color_threshold
    )
    plt.grid()

    plt.subplot(2, 1, 2)
    distances = linkage_matrix[:, 2][::-1]
    plt.plot(distances, "-o")
    plt.gca().invert_xaxis()
    plt.grid()
