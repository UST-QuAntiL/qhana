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
import copy
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
from qhana.backend.timer import Timer
import qhana.backend.scaling as scal
import qhana.backend.clustering as clu
import qhana.backend.dataForPlots as dfp
import qhana.backend.plotsForCluster as pfc
from qhana.backend.entity import Costume, CostumeFactory, Entity, EntityFactory
from qhana.backend.entityComparer import CostumeComparer, EmptyAttributeAction
from qhana.backend.attributeComparer import AttributeComparerType
from qhana.backend.aggregator import AggregatorType
from qhana.backend.elementComparer import ElementComparerType, WuPalmer
from qhana.backend.transformer import TransformerType
from qhana.backend.entityService import EntityService
import random
import qhana.backend.savingAndLoading as sal
from qhana.backend.entitySimilarities import EntitySimilarities
import concurrent.futures
from multiprocessing import Pool
from qhana.backend.entityService import Subset
from qhana.backend.classicNaiveMaxCutSolver import ClassicNaiveMaxCutSolver
from qhana.backend.sdpMaxCutSolver import SdpMaxCutSolver
import networkx as nx

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

def calc(node_pairs, tax):
    sims = {}
    comparer = WuPalmer()

    for node1, node2 in node_pairs:
        sim = comparer.compare(node1, node2, tax)
        sims[(node1, node2)] = sim

    return sims

def test3(command_args):
    db = Database()
    db.open()
    """
    tax = Taxonomie.create_from_db(TaxonomieType.charaktereigenschaft)
    graph = tax.graph
    tuples = graph.nodes.items()
    nodes = []
    amount = graph.number_of_nodes()
    # set custom amount
    amount = 40

    for name, value in tuples:
        nodes.append(name)

    print("Elements = " + str(amount))

    comparer = WuPalmer()

    # this shall be the dict with
    # { (taxonomie_entry_1, taxonomie_entry_2), value }
    base_parallel1 = {}
    base_parallel2 = {}
    base_serial = {}

    # create timer
    check: Timer = Timer()

    # starttimer
    check.start()

    # Z.B. Charaktereigenschaften
    # --> 800 möglichkeiten
    # --> ^2 => 800^2

    threads = 4
    taxs = []
    node_pairs = []
    node_pairs_thread = []
    inputs = []

    # make copy for each thread
    for t in range(0, threads):
        node_pairs_thread.append([])
        taxs.append(copy.deepcopy(tax))

    # get each thread his work
    for i in range(0, amount):
        for j in range(i, amount):
            node_pairs.append((nodes[i], nodes[j]))

    for i in range(0, len(node_pairs)):
        t = i % threads
        node_pairs_thread[t].append(node_pairs[i])

    # create inputs
    for t in range(0, threads):
        inputs.append((node_pairs_thread[t], taxs[t]))

    # print work per thread
    for t in range(0, threads):
        print("Thread " + str(t) + " has " + str(len(node_pairs_thread[t])) + " to do")

    # start task on threadpool
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = { executor.submit(calc, n, t): (n, t) for n, t in inputs }

    # get results
    for future in concurrent.futures.as_completed(futures):
        sims = future.result()
        base_parallel1.update(sims)

    # stop timer
    check.stop()

    # create timer
    check: Timer = Timer()

    # starttimer
    check.start()

    # start task on threadpool with multithreading (process based)
    pool = Pool(processes=threads)
    handles = []

    # start jobs
    for thread_input in inputs:
        handles.append(pool.apply_async(calc, (thread_input[0], thread_input[1])))

    # get results
    for handle in handles:
        sims = handle.get()
        base_parallel2.update(sims)

    # stop timer
    check.stop()

    # create timer
    check: Timer = Timer()

    # starttimer
    check.start()

    for i in range(0, amount):
        for j in range(i, amount):
            node1 = nodes[i]
            node2 = nodes[j]
            sim = comparer.compare(node1, node2, tax)
            base_serial[(node1, node2)] = sim

    # stop timer
    check.stop()

    diff1 = 0.0
    diff2 = 0.0

    for node1, node2 in base_serial.keys():
        diff1 += abs(base_serial[(node1, node2)] - base_parallel1[(node1, node2)])
        diff2 += abs(base_serial[(node1, node2)] - base_parallel2[(node1, node2)])

    print("The difference between serial and parallel1 is " + str(diff1))
    print("The difference between serial and parallel2 is " + str(diff2))

    return
    """
    # create the plan, i.e. specify which
    # attributes we want to have and also which
    # attribute comparer and element comparer
    # we want to choose for the specified attribute
    """
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
    """
    COSTUME_PLAN = [
        AggregatorType.mean,
        TransformerType.linearInverse,
    ]

    for attribute in Attribute:
        if  attribute == Attribute.dominantesAlter or \
            attribute == Attribute.kostuemZeit or \
            attribute == Attribute.alter or \
            attribute == Attribute.trageweise:
            continue
        COSTUME_PLAN.append(
            (
                attribute, 
                ElementComparerType.wuPalmer, 
                AttributeComparerType.singleElement, 
                EmptyAttributeAction.ignore
            )
        )

    # create the service
    service = EntityService()

    # add the plan: this will just
    # load the enums for attribute comparer
    # element comparer, transformer and aggregator, ...
    service.add_plan(COSTUME_PLAN)

    # Start stopping time
    check: Timer = Timer()
    check.start()
    
    # create the entities out of the database
    # 10 entities for example
    # here we create ALL entities, no filter
    amount = 10000
    service.create_entities(db, amount)

    entities = service.get_entities()

    # create the components, i.e. attribute comparer
    # element comparer ...
    # here, the objects are created
    service.create_components()

    #checkCache: Timer = Timer()
    #checkCache.start()

    #service.create_caches()
    #checkCache.stop()

    # create similarity matrix
    checkCache: Timer = Timer()
    for i in range(0, len(entities)):
        for j in range(0, len(entities)):
            #print(str(service.calculate_similarity(i, j)) + " ")
            service.calculate_similarity(i, j)
        print(str((i + 1) * amount) + " / " + str(amount * amount))
    check.stop()

    return

    # get a listof possible values in order
    # to let the user choose values for
    # filtering
    # NOTE: this needs to be done
    # after we vcalled create_entities()
    # as we load the values out of the loeaded
    # entities
    values_for_ortsbegebenheit = service.get_domain(Attribute.ortsbegebenheit)
    values_for_charaktereigenschaft = service.get_domain(Attribute.charaktereigenschaft)

    # print the values
    print("Possible values for Ortsbegebenheit:")
    print(values_for_ortsbegebenheit)
    print()
    print("Possible values for Charaktereigenschaft:")
    print(values_for_charaktereigenschaft)
    print()

    # add filter rule
    # every filter rule can be interpreted as
    # a logical AND operation, i.e.
    # ortsbegebenheit == "blabla" AND charaktereigenschaft == "blublu"

    ortsbegebenheit_filter_element = next(iter(values_for_ortsbegebenheit))
    print("Filter with Ortsbegebenheit = " + str(ortsbegebenheit_filter_element))
    print()

    charaktereigenschaft_filter_element = next(iter(values_for_charaktereigenschaft))
    print("Filter with Charaktereigenschaft = " + str(charaktereigenschaft_filter_element))
    print()

    #service.add_filter_rule(Attribute.ortsbegebenheit, ortsbegebenheit_filter_element)
    #service.add_filter_rule(Attribute.charaktereigenschaft, charaktereigenschaft_filter_element)

    # set seed before call get_entities()
    # therefore, we permutate the returned list
    service.set_seed(42)

    # get the list of entities if needed
    # this is not necessarily the case since
    # we can just compare entities using
    # their ids
    # however, we will here just get the filtered
    # entities
    entities = service.get_entities()

    len_entities = len(entities)

    for entity in entities:
        print(entity)
        print()

    # create timer
    check: Timer = Timer()

    print("Start elementwise comparsion with " + str(len_entities) + " entities")

    # starttimer
    check.start()

    # compare all 10 entities with each other:
    # at the moment, the ID of an entity is just
    # the number, i.e. the number in the array
    # they have been loaded out of the database.
    count = 0
    for i in range(0, len_entities):
        for j in range(0, len_entities):
            count += 1
            if count % (len_entities * len_entities * 0.1) == 0:
                print(str(int(count/(len_entities * len_entities) * 100)) + " % compared from " + str(len_entities * len_entities) + " comparsions")
            try:
                #sim = service.calculate_distance(i, j)
                sim = service.entitiyComparer.calculate_distance(entities[i], entities[j])
            except Exception as err:
                print("Exception!!")
                print(entities[i])
                print(entities[i].get_kostuem_url())
                raise err

    # stop timer
    check.stop()

    db.close()

    return

def test2(command_args):
    db = Database()
    db.open()

    COSTUME_PLAN = [
        AggregatorType.mean,
        TransformerType.linearInverse,
    ]

    for attribute in Attribute:
        if  attribute == Attribute.dominantesAlter or \
            attribute == Attribute.kostuemZeit or \
            attribute == Attribute.alter or \
            attribute == Attribute.trageweise:
            continue
        COSTUME_PLAN.append(
            (
                attribute, 
                ElementComparerType.wuPalmer, 
                AttributeComparerType.singleElement, 
                EmptyAttributeAction.ignore
            )
        )

    # create the service
    service = EntityService()

    # add the plan: this will just
    # load the enums for attribute comparer
    # element comparer, transformer and aggregator, ...
    service.add_plan(COSTUME_PLAN)

    # Start stopping time
    check: Timer = Timer()
    check.start()
    
    # create the entities out of the database
    # 10 entities for example
    # here we create ALL entities, no filter
    amount = 10000
    service.create_subset(Subset.subset10, db)

    entities = service.get_entities()

    for i in range(0, len(entities)):
        Logger.error(str(entities[i]))

def test(command_args):
    graph = nx.complete_graph(10)
    for (u, v) in graph.edges():
        graph.add_edge(u, v, weight = 1.0)

    solver = SdpMaxCutSolver(graph)
    (cutValue, cutEdges) = solver.solve()
    Logger.normal("CutVlaue = " + str(cutValue))
    Logger.normal("CutEdges = " + str(cutEdges))

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
    plt.savefig("GraphFullyConnected.png", format="PNG")
    plt.clf()

    for (u, v) in cutEdges:
        graph.remove_edge(u, v)

    nx.draw(graph, pos)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
    plt.savefig("GraphMaxCut.png", format="PNG")   
    plt.clf()

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
            Attribute.rollenrelevanz,
            ElementComparerType.wuPalmer,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
    ]
    

    # Establish connection to db
    db = Database()
    db.open()

    
    # built timer object
    check: Timer = Timer()    

    # built a similarity matrix
    simi = EntitySimilarities(COSTUME_PLAN,True,41,Subset.subset25)
    check.start()
    similarities = simi.create_matrix_limited(0,10)
    check.stop()
    print(similarities)
    
    
    # Multidimensional scaling
    #Object
    mds = scal.ScalingFactory.create(scal.ScalingType.mds)
    mds.set_max_iter(3000)
    mds.set_eps(1e-9)
    mds.set_dissimilarity("precomputed")
    mds.set_dimensions(2)

    pos = mds.scaling(similarities)
    stress = mds.stress_level()
    Logger.normal("Stress Level should be between 0 and 0.15")
    Logger.normal("Stress: " + str(stress))

    # clustering with optics
    new_cluster = clu.ClusteringFactory.create(clu.ClusteringType.vqeMaxCut)
    labels = new_cluster.create_cluster(pos, similarities)
    #quit()
    #new_cluster = clu.ClusteringFactory.create(clu.ClusteringType.optics)
    #labels = new_cluster.create_cluster(pos,similarities)

    # dfp_instance
    dfp_instance = dfp.DataForPlots(similarities, simi.get_last_sequenz(),None,pos, labels )

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

    #plt.figure(4)
    #G = gridspec.GridSpec(1, 1)
    #ax1 = plt.subplot(G[0, 0])
    #pfc.PlotsForCluster.costume_table_plot(dfp_instance, ax1)


    plt.figure(figsize=(10, 10))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    ax3 = plt.subplot(G[0, 2])
    ax4 = plt.subplot(G[1, :])

    
    pfc.PlotsForCluster.scaling_2d_plot(dfp_instance, ax2)
    pfc.PlotsForCluster.cluster_2d_plot(dfp_instance ,ax3)
    #pfc.PlotsForCluster.costume_table_plot(dfp_instance, ax4)
    pfc.PlotsForCluster.similarity_plot(dfp_instance, ax1)
    plt.tight_layout()
    
    plt.show()
    return

def main() -> None:
    parse()
    return

if __name__== "__main__":
    main()