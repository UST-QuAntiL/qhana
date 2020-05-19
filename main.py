import backend.elementComparer as elemcomp
import backend.attributeComparer as attrcomp
import backend.aggregator as aggre
import argparse
from backend.database import Database
from backend.taxonomie import Taxonomie, TaxonomieType
from backend.attribute import Attribute
from backend.logger import Logger, LogLevel
import numpy as np
import re
import csv
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
from backend.entity import Costume, CostumeFactory, Entity, EntityFactory
from backend.entityComparer import CostumeComparer, EmptyAttributeAction
from backend.attributeComparer import AttributeComparerType
from backend.aggregator import AggregatorType
from backend.elementComparer import ElementComparerType
from backend.transformer import TransformerType
from backend.entityService import EntityService
import random
import backend.savingAndLoading as sal
from backend.entitySimilarities import EntitySimilarities

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
    parser.add_argument('-db', '--database_config_file',
        dest='database_config_file',
        help='filepath for the *.ini file for the database connection',
        default='config.ini',
        required=False,
        type=str
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
        default='taxonomies',
        required=False,
        type=str
    )

    # Add parser for list_implemented_taxonomies
    list_implemented_taxonomies_parser = commands.add_parser('list_implemented_taxonomies',
        description='lists all the implemented taxonomies that can be used for machine learning')

    # Add parser for list_implemented_attributes
    list_implemented_attributes_parser = commands.add_parser('list_implemented_attributes',
        description='lists all the implemented attributes that can be used for machine learning')

    # Add parser for list_implemented_attribute_comparer
    list_implemented_attribute_comparer_parser = commands.add_parser('list_implemented_attribute_comparer',
        description='lists all the implemented attribute comparer that can be used for machine learning')

    # Add parser for list_implemented_aggregator
    list_implemented_aggregator_parser = commands.add_parser('list_implemented_aggregator',
        description='lists all the implemented aggregator that can be used for machine learning')

    # Add parser for list_implemented_element_comparer
    list_implemented_aggregator_parser = commands.add_parser('list_implemented_element_comparer',
        description='lists all the implemented element comparer that can be used for machine learning')

    # Add parser for list_implemented_transformer
    list_implemented_transformer_parser = commands.add_parser('list_implemented_transformer',
        description='lists all the implemented transformer that can be used for machine learning')

    # Add parser for export_csv
    export_csv_parser = commands.add_parser('export_csv',
        description='exports the loaded entities into csv file')
    export_csv_parser.add_argument('-o', '--output_file',
        dest='output_file',
        help='specifies the filename for the output [default: entities.csv]',
        default='entities.csv',
        required=False,
        type=str
    )
    export_csv_parser.add_argument('-n', '--amount',
        dest='amount',
        help='specifies the amount of entities [default: all]',
        default=2147483646,
        required=False,
        type=int
    )

    # Add parser for test
    validate_database_parser = commands.add_parser('test',
        description='just executes the things within test function')

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
    Database.config_file_default = args.database_config_file

    # Check which command is being used and run it
    if args.validate_database is not None:
        validate_database(args.validate_database)
    elif args.old_main is not None:
        old_main()
    elif args.create_taxonomies is not None:
        create_taxonomies(args.create_taxonomies)
    elif args.test is not None:
        test(args.test)
    elif args.list_implemented_taxonomies is not None:
        list_implemented_taxonomies(args.list_implemented_taxonomies)
    elif args.list_implemented_attributes is not None:
        list_implemented_attributes(args.list_implemented_attributes)
    elif args.list_implemented_attribute_comparer is not None:
        list_implemented_attribute_comparer(args.list_implemented_attribute_comparer)
    elif args.list_implemented_aggregator is not None:
        list_implemented_aggregator(args.list_implemented_aggregator)
    elif args.list_implemented_element_comparer is not None:
        list_implemented_element_comparer(args.list_implemented_element_comparer)
    elif args.list_implemented_transformer is not None:
        list_implemented_transformer(args.list_implemented_transformer)
    elif args.export_csv is not None:
        export_csv(args.export_csv)    
    else:
        Logger.normal("Wrong command. Please run -h for see available commands.")
    return

def validate_database(command_args):
    #db = Database()
    #db.open()
    #db.validate_all_costumes(command_args.output_file)
    return

def create_taxonomies(command_args):
    db = Database()
    db.open()

    for taxonomieType in TaxonomieType:
        tax = Taxonomie.create_from_db(taxonomieType, db)
        tax.save_json(directory=command_args.output_directory)
        tax.save_plot(directory=command_args.output_directory)

    return

def list_implemented_taxonomies(command_args):
    Logger.normal("The following taxonomies are currently available for machine learning")
    for taxonomie in TaxonomieType:
        Logger.normal(TaxonomieType.get_name(taxonomie))
    return

def list_implemented_attributes(command_args):
    Logger.normal("The following attributes are currently available for machine learning")
    for attribute in Attribute:
        Logger.normal(Attribute.get_name(attribute))
    return

def list_implemented_attribute_comparer(command_args):
    Logger.normal("The following attribute comparer are currently available for machine learning")
    for attributeComparer in AttributeComparerType:
        Logger.normal(str(attributeComparer) + ": " + AttributeComparerType.get_description(attributeComparer))
    return

def list_implemented_aggregator(command_args):
    Logger.normal("The following aggregator are currently available for machine learning")
    for aggregator in AggregatorType:
        Logger.normal(str(aggregator) + ": " + AggregatorType.get_description(aggregator))
    return

def list_implemented_element_comparer(command_args):
    Logger.normal("The following element comparer are currently available for machine learning")
    for elementComparer in ElementComparerType:
        Logger.normal(str(elementComparer) + ": " + ElementComparerType.get_description(elementComparer))
    return

def list_implemented_transformer(command_args):
    Logger.normal("The following transformer are currently available for machine learning")
    for transformer in TransformerType:
        Logger.normal(str(transformer) + ": " + TransformerType.get_description(transformer))
    return

def export_csv(command_args):
    db = Database()
    db.open()

    entities = EntityFactory.create(list(Attribute), db, command_args.amount)

    with open(command_args.output_file, 'w', newline='') as file:
        writer = csv.writer(file)

        all_attributes = list(Attribute)

        header = []
        for attribute in all_attributes:
            string = str(Attribute.get_name(attribute)).lower()
            string = string.replace("ä", "ae")
            string = string.replace("ü", "ue")
            string = string.replace("ö", "oe")
            string = string.replace("ß", "ss")
            string = string.replace(" ", "_")
            header.append(string)

        writer.writerow(header)

        for entity in entities:
            body = []
            for attribute in all_attributes:
                if len(entity.values[attribute]) == 0:
                    body.append("")
                else:
                    string = str(entity.values[attribute][0]).lower()
                    if string == "-":
                        string = ""
                    string = string.replace("ä", "ae")
                    string = string.replace("ü", "ue")
                    string = string.replace("ö", "oe")
                    string = string.replace("ß", "ss")
                    string = string.replace(" ", "_")
                    body.append(string)
            writer.writerow(body)
    return

def test(command_args):
    db = Database()
    db.open()

    # create the plan, i.e. specify which
    # attributes we want to have and also which
    # attribute comparer and element comparer
    # we want to choose for the specified attribute
    COSTUME_PLAN = [
        AggregatorType.mean,
        TransformerType.linearInverse,
        (
            Attribute.dominanteFarbe,
            ElementComparerType.wuPalmer,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.dominanteCharaktereigenschaft,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.dominanterZustand,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.stereotyp,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.geschlecht,
            ElementComparerType.wuPalmer,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.dominanterAlterseindruck,
            ElementComparerType.wuPalmer,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.genre, None, None, None
        ),
        (
            Attribute.kostuemZeit,
            ElementComparerType.timeTanh,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.rollenrelevanz,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
    ]
    
    # create the service
    service = EntityService()

    # add the plan: this will just
    # load the enums for attribute comparer
    # element comparer, transformer and aggregator, ...
    service.add_plan(COSTUME_PLAN)
    
    # create the entities out of the database
    # 10 entities for example
    amount = 10
    service.create_entities(db, amount)

    # create the components, i.e. attribute comparer
    # element comparer ...
    # here, the objects are created
    service.create_components()
    
    # get the list of entities if needed
    # this is not necessarily the case since
    # we can just compare entities using
    # their ids
    entities = service.get_entities()

    # compare all 10 entities with each other:
    # at the moment, the ID of an entity is just
    # the number, i.e. the number in the array
    # they have been loaded out of the database.
    for i in range(0, amount):
        for j in range(0, amount):
            sim = service.calculate_distance(i, j)
            print("Element " + str(i) + " <-> Element " + str(j) + " = " + str(sim))

    return

def print_costumes(costumes: [Costume]) -> None:
    for i in range(1, len(costumes)):
        print(str(i) + ": " + str(costumes[i]))

def old_main() -> None:


    COSTUME_PLAN = [
        AggregatorType.mean,
        TransformerType.linearInverse,
        (
            Attribute.dominanteFarbe,
            ElementComparerType.wuPalmer,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.dominanteCharaktereigenschaft,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.dominanterZustand,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.stereotyp,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.geschlecht,
            ElementComparerType.wuPalmer,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.dominanterAlterseindruck,
            ElementComparerType.wuPalmer,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.genre, None, None, None
        ),
        (
            Attribute.kostuemZeit,
            ElementComparerType.timeTanh,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ),
        (
            Attribute.rollenrelevanz,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
    ]
    

    file_folder_1 = "Versuch_1"

    saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.costumePlan)
    saf_cp.set(file_folder_1,COSTUME_PLAN)
    saf_cp.saving()

    del COSTUME_PLAN

    test = saf_cp.loading()
    COSTUME_PLAN = test.get_object()
    #print(str(COSTUME_PLAN))


    
    simi = EntitySimilarities(COSTUME_PLAN,True,20)
    simi.create_matrix_limited(0,15)
    file_folder_1 = "Versuch_1"

    saf_simi = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.entitySimilarities)
    saf_simi.set(file_folder_1,simi)
    saf_simi.saving()

    del simi

    test = saf_simi.loading()
    simi = test.get_object()
    check = Timer()
    check.start()
    similarities = simi.create_matrix_limited(4,18)
    check.stop()



    quit()
    # Establish connection to db
    db = Database()
    db.open()

    costumes = CostumeFactory.create(db, 41)
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
    simi = Similarities.only_costumes(costumes,True)
    check.start()
    simi.create_matrix_limited(0,40)
    check.stop()
    #Logger.normal("similarities")
    #Logger.normal(str(similarities))
    #costumes_simi: List[Costume] = simi.get_list_costumes()
    #for i in simi.get_last_sequenz():
    #    Logger.normal("index="+str(i)+ " : " +str(costumes_simi[i]))
    file_folder_1 = "Versuch_1"

    saf_simi = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.costumeSimilarities)
    saf_simi.set(file_folder_1,simi)
    saf_simi.saving()

    del simi

    test = saf_simi.loading()
    simi = test.get_object()
    check.start()
    similarities = simi.create_matrix_limited(0,40)
    check.stop()

    # Multidimensional scaling
    #Object
    mds = scal.ScalingFactory.create(scal.ScalingType.mds)
    mds.set_max_iter(3000)
    mds.set_eps(1e-9)
    mds.set_dissimilarity("precomputed")
    mds.set_dimensions(2)
    
    #mds.d2_plot(simi.get_last_sequenz(),simi.get_list_costumes())

    saf = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.scaling)
    saf.set(file_folder_1,mds)
    saf.saving()

    del mds

    test = saf.loading()
    mds = test.get_object()
    pos = mds.scaling(similarities)
    stress = mds.stress_level()
    Logger.normal("Stress Level should be between 0 and 0.15")
    Logger.normal("Stress: " + str(stress))

    # clustering with optics
    new_cluster = clu.ClusteringFactory.create(clu.ClusteringType.optics)
    labels = new_cluster.create_cluster(pos)

    saf_clu = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.clustering)
    saf_clu.set(file_folder_1,new_cluster)
    saf_clu.saving()

    del new_cluster

    test = saf_clu.loading()
    new_cluster = test.get_object()
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
    plt.figure(1)
    G = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(G[0, 0])
    pfc.PlotsForCluster.similarity_plot(dfp_instance, ax1)
    
    plt.figure(2)
    G = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(G[0, 0])
    pfc.PlotsForCluster.scaling_2d_plot(dfp_instance, ax1)

    plt.figure(3)
    G = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(G[0, 0])
    pfc.PlotsForCluster.cluster_2d_plot(dfp_instance ,ax1)

    plt.figure(4)
    G = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(G[0, 0])
    pfc.PlotsForCluster.costume_table_plot(dfp_instance, ax1)


    plt.figure(figsize=(10, 10))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    ax3 = plt.subplot(G[0, 2])
    ax4 = plt.subplot(G[1, :])

    
    pfc.PlotsForCluster.scaling_2d_plot(dfp_instance, ax2)
    pfc.PlotsForCluster.cluster_2d_plot(dfp_instance ,ax3)
    pfc.PlotsForCluster.costume_table_plot(dfp_instance, ax4)
    pfc.PlotsForCluster.similarity_plot(dfp_instance, ax1)
    plt.tight_layout()
    
    plt.show()
    return

def main() -> None:
    parse()
    return

if __name__== "__main__":
    main()