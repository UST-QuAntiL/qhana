"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from qiskit import Aer
from negativeRotation import NegativeRotation
from destructiveInterference import DestructiveInterference
from kmeansClusteringAlgorithm import standardize, normalize, generate_random_data
import matplotlib.pyplot as plt
import asyncio


def get_colors(k):
    """
    Return k colors in a list. We choose from 7 different colors.
    If k > 7 we choose colors more than once.
    """

    base_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    colors = []
    index = 1
    for i in range(0, k):
        if index % (len(base_colors) + 1) == 0:
            index = 1
        colors.append(base_colors[index - 1])
        index += 1
    return colors


def plot_data(data_lists, data_names, title, circle=False):
    """
    Plot all the data points with optional an unit circle.
    We expect to have data lists as a list of cluster points, i.e.
    data_lists =
    [ [ [x_0, y_0], [x_1, y_1], ... ], [x_0, y_0], [x_1, y_1], ... ] ]
    [      points of cluster 1       ,    points of cluster 2        ]
    data_names = [ "Cluster1", "Cluster2" ].
    """

    plt.clf()
    plt.figure(figsize=(7, 7), dpi=80)
    unit_circle_plot = plt.Circle((0, 0), 1.0, color='k', fill=False)
    ax = plt.gca()
    ax.cla()
    ax.set_xlim(-1.5, +1.5)
    ax.set_ylim(-1.5, +1.5)
    ax.set_title(title)
    if circle:
        ax.add_artist(unit_circle_plot)
    colors = get_colors(len(data_lists))
    for i in range(0, len(data_lists)):
        ax.scatter([data_points[0] for data_points in data_lists[i]],
                   [dataPoints[1] for dataPoints in data_lists[i]],
                   color=colors[i],
                   label=data_names[i])
    plt.show()
    return


def plot(data_raw, data, centroid_mapping, k):
    """
    Prepare data in order to plot.
    """

    data_texts = []
    clusters = dict()
    clusters_raw = dict()

    for i in range(0, centroid_mapping.shape[0]):
        cluster_number = int(centroid_mapping[i])
        if cluster_number not in clusters:
            clusters[cluster_number] = []
            clusters_raw[cluster_number] = []

        clusters[cluster_number].append(data[i])
        clusters_raw[cluster_number].append(data_raw[i])

    # add missing clusters that have no elements
    for i in range(0, k):
        if i not in clusters:
            clusters[i] = []

    clusters_plot = []
    clusters_raw_plot = []

    for i in range(0, k):
        clusters_plot.append([])
        clusters_raw_plot.append([])
        for j in range(0, len(clusters[i])):
            clusters_plot[i].append(clusters[i][j])
            clusters_raw_plot[i].append(clusters_raw[i][j])

    for i in range(0, k):
        data_texts.append("Cluster" + str(i))

    plot_data(clusters_plot, data_texts, "Preprocessed Data", circle=True)
    plot_data(clusters_raw_plot, data_texts, "Raw Data")


async def main():
    # define parameters
    amount_of_fake_data = 100

    backend = Aer.get_backend("qasm_simulator")
    max_qubits = 13
    shots_per_circuit = 8192
    k = 2
    max_runs = 10
    eps = 5

    data = generate_random_data(amount_of_fake_data)
    data_preprocessed = normalize(standardize(data))
    # algorithm = NegativeRotation(backend, max_qubits, shots_per_circuit, k, max_runs, eps, base_vector=[.5, .5])
    algorithm = DestructiveInterference(backend, max_qubits, shots_per_circuit, k, max_runs, eps, base_vector=[.5, .5])
    result = algorithm.perform_clustering(data)
    print(result)
    plot(data, data_preprocessed, result, k)


if __name__ == "__main__":
    asyncio.run(main())
