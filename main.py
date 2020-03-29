from backend.costume import Costume
import backend.elementComparer as elemcomp
import backend.attributeComparer as attrcomp
import backend.aggregator as aggre
import argparse
from backend.costumeComparer import CostumeComparer
from backend.database import Database
from backend.taxonomie import Taxonomie
from backend.attribute import Attribute
from backend.logger import Logger, LogLevel
import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
from backend.similarities import Similarities
from typing import List
import sys

# Used for creating the namespaces from parsing
def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args

def parse():
    parser = argparse.ArgumentParser()

    # Add global arguments
    parser.add_argument('-ll', '--log_level',
        dest='log_level',
        help='log level for the current session: 0 - nothing, 1 - errors [default], 2 - warnings, 3 - debug',
        default=1,
        required=False,
        type=int
    )

    # Add additional global arguments here, like above

    # Add commands entry in parser
    commands = parser.add_subparsers(title='commands')

    # Add parser for validate_database
    validate_database_parser = commands.add_parser('validate_database',
        description='check if the data is consistent with the taxonomies')
    validate_database_parser.add_argument('-o', '--output_file',
        dest='output_file',
        help='specifies the filename for the output [default: /log/<datetime>_db_validation.log]',
        default=None,
        required=False,
        type=str
    )

    # Add parser for others (just the other main routine)
    # This should not be needed in the future when
    # all commands are real commands with their own
    # parameters etc.
    validate_database_parser = commands.add_parser('old_main',
        description='runs the old main method (just for compatibility)')

    # Add additional commands arguments here, like above

    # Parse all the arguments and commands
    args = parse_args(parser, commands)
    
    # Set global arguments first
    Logger.initialize(args.log_level)

    # Check which command is being used and run it
    if args.validate_database is not None:
        validate_database(args.validate_database)
    elif args.old_main is not None:
        old_main()
    else:
        Logger.normal("Wrong command. Please run -h for see available commands.")

    return

def validate_database(command_args):
    db = Database()
    db.open()
    db.validate_all_costumes(command_args.output_file)
    return

def print_costumes(costumes: [Costume]) -> None:
    for i in range(1, len(costumes)):
        print(str(i) + ": " + str(costumes[i]))

def old_main() -> None:
    # Establish connection to db
    db = Database()
    db.open()

    costumes = db.get_costumes()
    costumeComparer = CostumeComparer()

    #print("Hier 1")
    #first = len(costumes)-1
    #second = 17
    #print("Hier 2")
    #comparedResult = costumeComparer.compare_distance(costumes[first], costumes[second])
    #print(costumes[first])
    #print(costumes[second])
    #print("Compared result: " + str(round(comparedResult, 2)))

    
    #built a similarity matrix
    simi =Similarities.only_costumes(costumes)
    similarities = simi.create_matrix_limited(0,20)
    Logger.normal("similarities")
    Logger.normal("\n"+str(similarities))
    costumes_simi: List[Costume] = simi.get_list_costumes()
    #for i in simi.get_last_sequenz():
    #    print("index="+str(i)+ " : " +str(costumes_simi[i]))
    
    # Multidimensional scaling
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_
    Logger.normal("Position eukl.")
    Logger.normal("\n" + str(pos))
    stress = mds.fit(similarities).stress_
    Logger.normal("Stress Level should be between 0 and 0.15")
    Logger.normal("Stress: " + str(round(stress, 4)))
    # plot multidimensional scaling positions
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])

    s = 100
    plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    EPSILON = np.finfo(np.float32).eps
    #print("similarities.max")
    #print(similarities.max())
    similarities = similarities.max() / (similarities + EPSILON) * 100
    #print("similarities after max/(sim+eps)*100")
    #print(similarities)
    np.fill_diagonal(similarities, 0)
    # Plot the edges
    #start_idx, end_idx = np.where(pos)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[pos[i, :], pos[j, :]]
                for i in range(len(pos)) for j in range(len(pos))]
    #print("segments")
    #print(segments)
    values = np.abs(similarities)
    #print("Values")
    #print(values)
    lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.Blues,
                    norm=plt.Normalize(0, values.max()))
    lc.set_array(similarities.flatten())
    lc.set_linewidths(np.full(len(segments), 0.5))
    ax.add_collection(lc)
    # describe points
    style = dict(size=7, color='black')
    count: int = 0
    for i in simi.get_last_sequenz():
        txt = str(i)+". " +str(costumes_simi[i])
        txt = re.sub("(.{20})", "\\1-\n", str(txt), 0, re.DOTALL)
        plt.annotate(txt, (pos[count, 0], pos[count, 1]), **style)
        count += 1
    #plt.ylim((-0.6,0.6))
    #plt.xlim((-0.5,0.5))

    

    # clustering with optics
    clust = OPTICS(min_samples=3, xi=.05, min_cluster_size=.05)
    clust.fit(pos)

    # plot things 
    
    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
    labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

    space = np.arange(len(pos))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = pos[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.plot(pos[clust.labels_ == -1, 0], pos[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 0.5
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = pos[labels_050 == klass]
        ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    ax3.plot(pos[labels_050 == -1, 0], pos[labels_050 == -1, 1], 'k+', alpha=0.1)
    ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = pos[labels_200 == klass]
        ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax4.plot(pos[labels_200 == -1, 0], pos[labels_200 == -1, 1], 'k+', alpha=0.1)
    ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

    plt.tight_layout()
    plt.show()
    return

def main() -> None:
    parse()
    return

if __name__== "__main__":
    main()