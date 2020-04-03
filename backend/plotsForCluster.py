import backend.dataForPlots as dfp
from typing import List
import numpy as np
from backend.costume import Costume
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import re
from matplotlib.collections import LineCollection
from backend.logger import Logger, LogLevel
import pandas as pd

class PlotsForCluster():

    @staticmethod
    def scaling_2d_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        similarity_matrix: np.matrix = dataforplots.get_similarity_matrix() 
        position_matrix: np.matrix = dataforplots.get_position_matrix()
        last_sequenz: List[int] = dataforplots.get_sequenz()
        costumes: List[Costume] = dataforplots.get_list_costumes()
        
        if len(position_matrix[0]) != 2:
            Logger.error("Dimension of Position Matrix is not 2!")
            raise Exception("Dimension of Position Matrix is not 2!")

        
        #ax = subplot.axes([0., 0., 1., 1.])
        s = 100
        subplot.scatter(position_matrix[:, 0], position_matrix[:, 1], color='turquoise', s=s, lw=0, label='MDS')
        subplot.legend(scatterpoints=1, loc='best', shadow=False)
        EPSILON = np.finfo(np.float32).eps
        #print("similarities.max")
        #print(similarities.max())
        similarity_matrix = similarity_matrix.max() / (similarity_matrix + EPSILON) * 100
        #print("similarities after max/(sim+eps)*100")
        #print(similarities)
        np.fill_diagonal(similarity_matrix, 0)
        # Plot the edges
        #start_idx, end_idx = np.where(pos)
        # a sequence of (*line0*, *line1*, *line2*), where::
        #            linen = (x0, y0), (x1, y1), ... (xm, ym)
        segments = [[position_matrix[i, :], position_matrix[j, :]]
                    for i in range(len(position_matrix)) for j in range(len(position_matrix))]
        #print("segments")
        #print(segments)
        values = np.abs(similarity_matrix)
        #print("Values")
        #print(values)
        lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.Blues,
                        norm=plt.Normalize(0, values.max()))
        lc.set_array(similarity_matrix.flatten())
        lc.set_linewidths(np.full(len(segments), 0.5))
        #ax.add_collection(lc)
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
        #subplot.draw()


    @staticmethod
    def cluster_2d_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        #similarity_matrix: np.matrix = dataforplots.get_similarity_matrix() 
        position_matrix: np.matrix = dataforplots.get_position_matrix()
        sequenz: List[int] = dataforplots.get_sequenz()
        #costumes: List[Costume] = dataforplots.get_list_costumes()
        labels: np.matrix = dataforplots.get_labels()

        if len(position_matrix[0]) != 2:
            Logger.error("Dimension of Position Matrix is not 2!")
            raise Exception("Dimension of Position Matrix is not 2!")

        colors = ['go', 'ro', 'bo', 'yo', 'co']
        for klass, color in zip(range(0, 5), colors):
            Xk = position_matrix[labels == klass]
            subplot.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.5)
        subplot.plot(position_matrix[labels == -1, 0], position_matrix[labels == -1, 1], 'k+', alpha=0.1)
        subplot.set_title('Automatic Clustering\nOPTICS')
        count: int = 0
        style = dict(size=7,color='black')
        for i in sequenz:
            #txt = str(i)+". " +str(costumes[i])
            txt = str(i)+". "
            txt = re.sub("(.{20})", "\\1-\n", str(txt), 0, re.DOTALL)
            subplot.annotate(txt, (position_matrix[count, 0], position_matrix[count, 1]), **style)
            count += 1

    @staticmethod
    def similarity_2d_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        similarity_matrix: np.matrix = dataforplots.get_similarity_matrix() 
        #position_matrix: np.matrix = dataforplots.get_position_matrix()
        sequenz: List[int] = dataforplots.get_sequenz()
        #costumes: List[Costume] = dataforplots.get_list_costumes()
        #labels: np.matrix = dataforplots.get_labels()
        cmap = cm.get_cmap('jet')
        cax = subplot.matshow(similarity_matrix, interpolation='nearest' , cmap=cmap)
        subplot.grid(True)
        subplot.set_title("Similarity Matrix")
        subplot.set_xticks(range(len(similarity_matrix)))
        subplot.set_xticklabels(sequenz, rotation=90)
        subplot.set_yticks(range(len(similarity_matrix)))
        subplot.set_yticklabels(sequenz)
        plt.colorbar(cax , ax=subplot, ticks=[0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


    @staticmethod
    def costume_table_plot(dataforplots: dfp.DataForPlots, subplot) -> None:
        #similarity_matrix: np.matrix = dataforplots.get_similarity_matrix() 
        #position_matrix: np.matrix = dataforplots.get_position_matrix()
        sequenz: List[int] = dataforplots.get_sequenz()
        costumes: List[Costume] = dataforplots.get_list_costumes()
        labels: np.matrix = dataforplots.get_labels()
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
        
        colColour: List = []
        inlist: List = []
        colors = ['#98FB98', '#FF7256', '#1E90FF', '#FFFF00', '#00F5FF']
        for i in range(len(labels)):
            if labels[i] == -1:
                inlist = []
                for j in range(len(df.columns)):
                    inlist.append('#F8F8FF')
                colColour.append(inlist)
            else:
                inlist = []
                for j in range(len(df.columns)):
                    inlist.append(colors[labels[i]])
                colColour.append(inlist)

        cell_text: [] = []
        for row in range(len(df)):
            #print(row)
            #print(df.iloc[row])
            cell_text.append(df.iloc[row])
        
        #,colWidths=[0.025,0.112, 0.16, 0.08,0.40,0.07,0.1,0.30]
        table = subplot.table(  cellText    =   cell_text,
                                cellColours =   colColour,
                                colColours  =   ['#76D7C4'] * len(df.columns), 
                                colLabels   =   df.columns,
                                cellLoc     =   "center", 
                                loc         =   'center',
                                bbox        =   [0, 0, 1, 1]
                                )
        table.auto_set_font_size(True)
        #table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        #print(list(range(len(df.columns))))
        #table.auto_set_column_width([1,1,1,1,1,1,1,1]) 
        #plt.subplots_adjust(right=0.8)
        subplot.axis('off')