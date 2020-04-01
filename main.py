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
from backend.timer import Timer
import backend.scaling as scal
import backend.clustering as clu
import backend.dataForPlots as dfp
import backend.plotsForCluster as pfc

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

    # Add parser for create_taxonomies
    validate_database_parser = commands.add_parser('create_taxonomies',
        description='creates the taxonomies from the database (svg, json and dot)')
    validate_database_parser.add_argument('-o', '--output_directory',
        dest='output_directory',
        help='specifies the directory for the output [default: /taxonomies]',
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
    elif args.create_taxonomies is not None:
        create_taxonomies(args.create_taxonomies)
    else:
        Logger.normal("Wrong command. Please run -h for see available commands.")

    return

def validate_database(command_args):
    db = Database()
    db.open()
    db.validate_all_costumes(command_args.output_file)
    return

def create_taxonomies(command_args):
    tax = Taxonomie()
    tax.load_all_from_database()
    tax.safe_all_to_file()
    tax.plot_all(display=False)
    return

def print_costumes(costumes: [Costume]) -> None:
    for i in range(1, len(costumes)):
        print(str(i) + ": " + str(costumes[i]))

def old_main() -> None:
    # Establish connection to db
    
    db = Database()
    db.open()

    costumes = db.get_costumes()
    #costumeComparer = CostumeComparer()

    #print("Hier 1")
    #first = len(costumes)-1
    #second = 17
    #print("Hier 2")
    #comparedResult = costumeComparer.compare_distance(costumes[first], costumes[second])
    #print(costumes[first])
    #print(costumes[second])
    #print("Compared result: " + str(round(comparedResult, 2)))

    # built timer object
    check: Timer = Timer()    

    # built a similarity matrix
    simi =Similarities.only_costumes(costumes,True)
    check.start()
    similarities = simi.create_matrix_limited(0,40)
    check.stop()
    Logger.normal("similarities")
    Logger.normal(str(similarities))
    #costumes_simi: List[Costume] = simi.get_list_costumes()
    #for i in simi.get_last_sequenz():
    #    Logger.normal("index="+str(i)+ " : " +str(costumes_simi[i]))
    
    

    # Multidimensional scaling
    #Object
    mds = scal.ScalingFactory.create(scal.ScalingType.mds)
    mds.set_max_iter(3000)
    mds.set_eps(1e-9)
    mds.set_dissimilarity("precomputed")
    mds.set_dimensions(2)
    pos = mds.scaling(similarities)
    Logger.normal("Position eukl.")
    Logger.normal(str(pos))
    stress = mds.stress_level()
    Logger.normal("Stress Level should be between 0 and 0.15")
    Logger.normal("Stress: " + str(stress))
    #mds.d2_plot(simi.get_last_sequenz(),simi.get_list_costumes())

    


    # clustering with optics
    new_cluster = clu.ClusteringFactory.create(clu.ClusteringType.optics)
    labels = new_cluster.create_cluster(pos)
    
    """
        Logger.error("--------Tests-------sollten noch getestet werden ---------------")
        new_cluster = clu.ClusteringFactory.create(clu.ClusteringType.optics)
    
        new_cluster.set_min_samples()
        new_cluster.set_max_eps()
        new_cluster.set_metric()
        new_cluster.set_p()
        new_cluster.set_metric_params()
        new_cluster.set_cluster_method()
        new_cluster.set_eps()
        new_cluster.set_xi()
        new_cluster.set_predecessor_correction()
        new_cluster.set_min_cluster_size()
        new_cluster.set_algorithm()
        new_cluster.set_leaf_size()
        new_cluster.set_n_jobs()

        Logger.error("Comparing with getter methodes")
        print(new_cluster.get_min_samples() == new_cluster.get_cluster_instance().min_samples)
        print(new_cluster.get_max_eps()== new_cluster.get_cluster_instance().max_eps)
        print(new_cluster.get_metric()== new_cluster.get_cluster_instance().metric)
        print(new_cluster.get_p()== new_cluster.get_cluster_instance().p)
        print(new_cluster.get_metric_params()== new_cluster.get_cluster_instance().metric_params)
        print(new_cluster.get_cluster_method()== new_cluster.get_cluster_instance().cluster_method)
        print(new_cluster.get_eps()== new_cluster.get_cluster_instance().eps)
        print(new_cluster.get_xi()== new_cluster.get_cluster_instance().xi)
        print(new_cluster.get_predecessor_correction()== new_cluster.get_cluster_instance().predecessor_correction)
        print(new_cluster.get_min_cluster_size()== new_cluster.get_cluster_instance().min_cluster_size)
        print(new_cluster.get_algorithm()== new_cluster.get_cluster_instance().algorithm)
        print(new_cluster.get_leaf_size()== new_cluster.get_cluster_instance().leaf_size)
        print(new_cluster.get_n_jobs()== new_cluster.get_cluster_instance().n_jobs)

        print(new_cluster.get_cluster_instance().min_samples)
        quit()
    """
    # dfp_instance
    dfp_instance = dfp.DataForPlots(similarities, simi.get_last_sequenz(),simi.get_list_costumes(),pos, labels )

    # plot things 

    plt.figure(figsize=(10, 10))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    ax3 = plt.subplot(G[0, 2])
    ax4 = plt.subplot(G[1, :])

    
    pfc.PlotsForCluster.scaling_2d_plot(dfp_instance, ax2)
    pfc.PlotsForCluster.cluster_2d_plot(dfp_instance ,ax3)
    pfc.PlotsForCluster.costume_table_plot(dfp_instance, ax4)
    pfc.PlotsForCluster.similarity_2d_plot(dfp_instance, ax1)
    #plt.tight_layout()
    
    plt.show()
    return

def main() -> None:
    parse()
    return

if __name__== "__main__":
    main()