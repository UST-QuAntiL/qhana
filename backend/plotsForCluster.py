import backend.dataForPlots as dfp
from typing import List
import numpy as np
from backend.costume import Costume
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as cl
import re
from matplotlib.collections import LineCollection
from backend.logger import Logger, LogLevel
import pandas as pd

class PlotsForCluster():

    @staticmethod
    def scaling_2d_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        """
        Plot for Cluster only 2d Plot 
        """
        # getter from dfp
        similarity_matrix: np.matrix = dataforplots.get_similarity_matrix() 
        position_matrix: np.matrix = dataforplots.get_position_matrix()
        last_sequenz: List[int] = dataforplots.get_sequenz()
        costumes: List[Costume] = dataforplots.get_list_costumes()
        
        # check if position matrix have only 2 dimensions
        if len(position_matrix[0]) != 2:
            Logger.error("Dimension of Position Matrix is not 2!")
            raise Exception("Dimension of Position Matrix is not 2!")

        # set point width
        s = 100

        # set 2d for scaling
        # plot the nodes
        subplot.scatter(position_matrix[:, 0], position_matrix[:, 1], color='turquoise', s=s, lw=0, label='MDS')
        subplot.legend(scatterpoints=1, loc='best', shadow=False)
        # Plot the edges
        EPSILON = np.finfo(np.float32).eps
        similarity_matrix = similarity_matrix.max() / (similarity_matrix + EPSILON) * 100
        np.fill_diagonal(similarity_matrix, 0)
        segments = [[position_matrix[i, :], position_matrix[j, :]]
                    for i in range(len(position_matrix)) for j in range(len(position_matrix))]
        values = np.abs(similarity_matrix)
        lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.Blues,
                        norm=plt.Normalize(0, values.max()))
        lc.set_array(similarity_matrix.flatten())
        lc.set_linewidths(np.full(len(segments), 0.5))
        subplot.add_collection(lc)
        
        # describe points
        style = dict(size=7, color='black')
        count: int = 0
        for i in last_sequenz:
            #txt = str(i)+". " +str(costumes[i])
            txt = str(i)+". "
            txt = re.sub("(.{20})", "\\1-\n", str(txt), 0, re.DOTALL)
            subplot.annotate(txt, (position_matrix[count, 0], position_matrix[count, 1]), **style)
            count += 1
        subplot.set_title('Multidimension Scaling \n')
        #subplot.ylim((-0.6,0.6))
        #subplot.xlim((-0.5,0.5))

    @staticmethod
    def cluster_2d_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        """
        Plot for Cluster only 2d Plot 
        """
        # getter from dfp
            #similarity_matrix: np.matrix = dataforplots.get_similarity_matrix() 
            #costumes: List[Costume] = dataforplots.get_list_costumes()
        position_matrix: np.matrix = dataforplots.get_position_matrix()
        sequenz: List[int] = dataforplots.get_sequenz()
        labels: np.matrix = dataforplots.get_labels()
        
        # set colormap and max and min color 
        cmap: cl.LinearSegmentedColormap = cm.get_cmap('jet')
        min_color: int = 0.3
        max_color: int = 0.85

        # check if position matrix have only 2 dimensions
        if len(position_matrix[0]) != 2:
            Logger.error("Dimension of Position Matrix is not 2!")
            raise Exception("Dimension of Position Matrix is not 2!")

        # set norm for range in colormap
        norm: cl.Normalize = cl.Normalize(vmin=-min_color/(max_color-min_color), vmax=(1-min_color)/(max_color-min_color))
        
        # set List for colors in cluster
        max_value: int = max(labels)
        colors: List = []
        for i in range(max_value+1):
            if i == 0 and max_value == 0:
                colors.append(cmap(norm(0)))
            else:    
                colors.append(cmap(norm(i/max_value)))
        
        # set plot for 2d clustering
        for klass, color in zip(range(0, max_value+1), colors):
            Xk = position_matrix[labels == klass]
            subplot.plot(Xk[:, 0], Xk[:, 1], color=color, marker='o', linestyle='None')
        subplot.plot(position_matrix[labels == -1, 0], position_matrix[labels == -1, 1], 'k+', alpha=0.1)
        subplot.set_title('Clustering')
        count: int = 0
        style = dict(size=7,color='black')
        for i in sequenz:
            #txt = str(i)+". " +str(costumes[i])
            txt = str(i)+". "
            txt = re.sub("(.{20})", "\\1-\n", str(txt), 0, re.DOTALL)
            subplot.annotate(txt, (position_matrix[count, 0], position_matrix[count, 1]), **style)
            count += 1

    @staticmethod
    def similarity_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        """
        plot the similarity plot 
        """
        # getter from dfp
            #position_matrix: np.matrix = dataforplots.get_position_matrix()
            #costumes: List[Costume] = dataforplots.get_list_costumes()
            #labels: np.matrix = dataforplots.get_labels()
        similarity_matrix: np.matrix = dataforplots.get_similarity_matrix() 
        sequenz: List[int] = dataforplots.get_sequenz()

        # set colormap
        cmap = cm.get_cmap('jet')

        # set plot for similarity plot
        cax = subplot.matshow(similarity_matrix, interpolation='nearest' , cmap=cmap)
        subplot.grid(True)
        subplot.set_title("Similarity Matrix")
        subplot.set_xticks(range(len(similarity_matrix)))
        subplot.set_xticklabels(sequenz, rotation=90)
        subplot.set_yticks(range(len(similarity_matrix)))
        subplot.set_yticklabels(sequenz)
        plt.colorbar(cax , ax=subplot, ticks=[0.0 ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    @staticmethod
    def costume_table_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        """
        table for plots with costume number
        """
        #getter from dfp
            #similarity_matrix: np.matrix = dataforplots.get_similarity_matrix() 
            #position_matrix: np.matrix = dataforplots.get_position_matrix()
        sequenz: List[int] = dataforplots.get_sequenz()
        costumes: List[Costume] = dataforplots.get_list_costumes()
        labels: np.matrix = dataforplots.get_labels()
        
        # set colormap and max and min color 
        cmap: cl.LinearSegmentedColormap = cm.get_cmap('jet')
        min_color: int = 0.3
        max_color: int = 0.85        
        
        # set dataframe for plot
        data: [] = []
        dominant_color: str = ""
        dominant_traits: [str] = []
        dominant_condition: str = ""
        stereotypes: [str] = []
        gender: str = ""
        dominant_age_impression: str = ""
        genres: [str] = []
        costume: int = 0
        for i in sequenz:
            costume = i
            dominant_color = costumes[i].dominant_color
            dominant_traits = costumes[i].dominant_traits
            dominant_condition= costumes[i].dominant_condition
            stereotypes = costumes[i].stereotypes
            gender = costumes[i].gender
            dominant_age_impression = costumes[i].dominant_age_impression
            genres = costumes[i].genres
            data.append([costume,dominant_color,dominant_traits, dominant_condition, stereotypes, gender, dominant_age_impression, genres])
        df = pd.DataFrame(data, columns=['Nr.','Farben','Charaktereigenschaft', 'Zustand', 'Stereotyp', 'Geschlecht', 'Alterseindruck', 'Genre' ])
        
        # set norm for range in colormap
        norm: cl.Normalize = cl.Normalize(vmin=-min_color/(max_color-min_color), vmax=(1-min_color)/(max_color-min_color))
        
        # set List for colors in cluster
        max_value: int = max(labels)
        colors: List = []
        for i in range(max_value+1):
            if i == 0 and max_value == 0:
                colors.append(cmap(norm(0)))
            else:    
                colors.append(cmap(norm(i/max_value)))

        # set color matrix for single cells in table
        colColour: List = []
        inlist: List = []
        for i in range(len(labels)):
            if labels[i] == -1:
                inlist = []
                for _ in range(len(df.columns)):
                    inlist.append('w')
                colColour.append(inlist)
            else:
                inlist = []
                for _ in range(len(df.columns)):
                    inlist.append(colors[labels[i]])
                colColour.append(inlist)

        # set cell text for table
        cell_text: [] = []
        for row in range(len(df)):
            cell_text.append(df.iloc[row])
        
        # plot table 
        subplot.axis('off')
        table = subplot.table(  cellText    =   cell_text,
                                cellColours =   colColour,
                                colColours  =   ['#76D7C4'] * len(df.columns), 
                                colLabels   =   df.columns,
                                cellLoc     =   "center", 
                                loc         =   'center',
                                bbox        =   [0, 0, 1, 1]
                                )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        