import backend.dataForPlots as dfp
from typing import List
import numpy as np
from backend.entity import Costume
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as cl
import re
from matplotlib.collections import LineCollection
from backend.logger import Logger, LogLevel
import pandas as pd
from backend.attribute import Attribute
from math import ceil
from sklearn import metrics
from backend.classification import get_subsetLabels

class PlotsForClassification():

    @staticmethod
    def classifier_2d_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        position_matrix: np.matrix = dataforplots.get_position_matrix()
        position_matrix_orig: np.matrix = dataforplots.get_position_matrix_orig()
        support_vectors: np.matrix = dataforplots.get_support_vectors()
        decision_fun = dataforplots.get_decision_fun()
        transform = dataforplots.get_transform2d()
        inverse_transform = dataforplots.get_inverse_transform()

        """ plot samples for 2 classes (fixed as in subsets) """
        n_samples = len(position_matrix)
        labels = get_subsetLabels(position_matrix_orig)

        subplot.scatter(position_matrix[:ceil(n_samples/2), 0], position_matrix[:ceil(n_samples/2), 1],
                        color='red', s=100, lw=0, label='1')
        subplot.scatter(position_matrix[ceil(n_samples/2):, 0], position_matrix[ceil(n_samples/2):, 1],
                        color='blue', s=100, lw=0, label='-1')

        """ circle support vectors """
        if len(support_vectors) > 0:
            support_vectors2d = transform(support_vectors)
            subplot.scatter(support_vectors2d[:,0], support_vectors2d[:,1],
                        s=150, linewidth=.5, facecolors='none', edgecolors='green', label="supp. vectors")

        """ draw decision fun boundaries """
        # get axes
#         xlim = np.array([np.min(position_matrix_orig[:,0]), np.max(position_matrix_orig[:,0])])*1.1
#         ylim = np.array([np.min(position_matrix_orig[:,1]), np.max(position_matrix_orig[:,1])])*1.1
        xlim = np.array([np.min(position_matrix[:,0]), np.max(position_matrix[:,0])])*1.1
        ylim = np.array([np.min(position_matrix[:,1]), np.max(position_matrix[:,1])])*1.1

        res = 50
        # generate a grid of res^n_axes points on the canvas
        x_linspace = np.linspace(xlim[0], xlim[1], res)
        y_linspace = np.linspace(ylim[0], ylim[1], res)

        y_grid, x_grid = np.meshgrid(y_linspace, x_linspace)
        grid = np.vstack([x_grid.ravel(), y_grid.ravel()])

        grid2d = grid.T #transform(grid.T)
        grid = inverse_transform(grid.T)
        #print(grid2d)

        # evaluate decision function for each point in the grid
        z_labels = decision_fun(grid)
        z_labels = z_labels.reshape(x_grid.shape)

        """ compute accuracy """
        predictions = decision_fun(position_matrix_orig)

        # draw contours
        subplot.contourf(grid2d[:,0].reshape(x_grid.shape), grid2d[:,1].reshape(x_grid.shape),
                        z_labels, #colors="k",
                        levels=1, linestyles=["-"],
                        cmap='winter', alpha=0.3)
        #subplot.contour(grid2d[:,0].reshape(x_grid.shape), grid2d[:,1].reshape(x_grid.shape),
        #                z_labels, colors="k",
        #                levels=1, linewidths=0.5, linestyles=["-"])

        accuracy = metrics.accuracy_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions)

        """ add legend """
        subplot.legend(scatterpoints=1, loc='best', shadow=False)
        subplot.set_title('Classification \naccuracy={}, precision={}, recall={}'.format(accuracy, precision, recall))

