import backend.dataForPlots as dfp
import numpy as np
from matplotlib import cm
from backend.logger import Logger, LogLevel
from backend.attribute import Attribute
from math import ceil
from sklearn import metrics

class PlotsForClassification():

    @staticmethod
    def classifier_2d_plot(dataforplots: dfp.DataForPlots, train_set, train_labels, test_set, test_labels, dict_label_class: dict, subplot) -> None:
        Logger.debug("Start plotting trained classifier.")
        position_matrix: np.matrix = dataforplots.get_position_matrix()
        position_matrix_orig: np.matrix = dataforplots.get_position_matrix_orig()
        support_vectors: np.matrix = dataforplots.get_support_vectors()
        decision_fun = dataforplots.get_decision_fun()
        transform = dataforplots.get_transform2d()
        inverse_transform = dataforplots.get_inverse_transform()

        """ create a dictionary for coloring the plot """
        classes = list(set(list(train_labels)+list(test_labels)))
        n_classes = len(classes)
        colors = cm.get_cmap("rainbow", n_classes)
        colors_dict = {}
        for i in range(n_classes):
            colors_dict[classes[i]] = colors(i)

        """ plot samples for 2 classes """
        train_set2d = transform(train_set)
        Logger.debug("Scatter train set.")
        for i in range(len(train_labels)):
            subplot.scatter(train_set2d[i][0], train_set2d[i][1],
                        color=colors_dict[train_labels[i]],
                        s=100, lw=0, label=dict_label_class[train_labels[i]])

        if not str(train_set) == str(test_set):
            Logger.debug("Scatter test set.")
            test_set2d = transform(test_set)
            for i in range(len(test_labels)):
                subplot.scatter(test_set2d[i][0], test_set2d[i][1],
                            color=colors_dict[test_labels[i]],
                            s=100, lw=0, label=dict_label_class[test_labels[i]])

        """ circle support vectors """
        if len(support_vectors) > 0:
            Logger.debug("Scatter support vectors.")
            support_vectors2d = transform(support_vectors)
            subplot.scatter(support_vectors2d[:,0], support_vectors2d[:,1],
                        s=150, linewidth=.5, facecolors='none', edgecolors='green', label="supp. vectors")

        """ mark train data """
        Logger.debug("Mark train data.")
        subplot.scatter(train_set2d[:,0], train_set2d[:,1], s=50, marker="x", label="train data")

        """ draw decision fun boundaries """
        Logger.debug("Generate grid.")
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
        Logger.debug("Evaluate decision function on grid.")
        z_labels = decision_fun(grid)
        z_labels = z_labels.reshape(x_grid.shape)

        # draw contours
        Logger.debug("Draw contours.")
        subplot.contourf(grid2d[:,0].reshape(x_grid.shape), grid2d[:,1].reshape(x_grid.shape),
                        z_labels, #colors="k",
                        levels=n_classes-1, linestyles=["-"],
                        cmap='winter', alpha=0.3)
        #subplot.contour(grid2d[:,0].reshape(x_grid.shape), grid2d[:,1].reshape(x_grid.shape),
        #                z_labels, colors="k",
        #                levels=1, linewidths=0.5, linestyles=["-"])

        """ compute accuracy """
        Logger.debug("Compute precision, accuracy, and recall")
        predictions = decision_fun(test_set)
        accuracy = round(metrics.accuracy_score(test_labels, predictions), 3)
        recall = round(metrics.recall_score(test_labels, predictions, average='micro' if n_classes > 2 else 'binary'), 3)
        precision = round(metrics.precision_score(test_labels, predictions, average='micro' if n_classes > 2 else 'binary'), 3)

        """ add legend """
        handles, labels = subplot.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        subplot.legend(by_label.values(), by_label.keys())

        #subplot.legend(scatterpoints=1, loc='best', shadow=False)
        subplot.set_title('Classification \naccuracy={}, precision={}, recall={}'.format(accuracy, precision, recall))
        Logger.debug("Finished plot.")