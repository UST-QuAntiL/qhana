from flask import Flask, render_template, url_for, session, send_file, request, flash , redirect
import threading, webbrowser
from backend.attribute import Attribute
from backend.taxonomie import TaxonomieType, Taxonomie
from backend.logger import Logger
from backend.database import Database
import os
from os.path import basename 
from mysql.connector import Error
import shutil
import zipfile
import io
import pathlib
from backend.aggregator import AggregatorType
from backend.transformer import TransformerType
from backend.attributeComparer import AttributeComparerType
from backend.elementComparer import ElementComparerType
from backend.entityComparer import EmptyAttributeAction
import pickle 
import backend.savingAndLoading as sal
from backend.entitySimilarities import EntitySimilarities
import backend.dataForPlots as dfp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import backend.plotsForCluster as pfc
import numpy as np
from backend.scaling import ScalingType, ScalingFactory , Scaling , MultidimensionalScaling
from backend.clustering import ClusteringType, ClusteringFactory, Clustering, Optics
import numpy as np

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.entitySimilarities: EntitySimilarities = None
app.scaling: Scaling = None
app.clustering: Clustering = None
app.strCostumePlan: list = None
app.result: list = None
app.tablelist: list = None
app.start: bool = True

@app.route("/home")
@app.route("/")
def home():
    if app.start :
        session["saveload"] = ""
        initialize_costumeplan()


    attributes = []
    for attribute in Attribute:
        attributes.append(Attribute.get_name(attribute))
    
    try:
        db = Database()
        db.open()
        dbConnectionString = "Database: Host=" + db.host + ", DB-Name=" + db.databaseName + ", User=" + db.user
    except Error as error:
        dbConnectionString =  str(error)

    session["dbConnectionString"] = dbConnectionString

    return render_template("home.html", attributes=attributes)

@app.route("/reset_all")
def reset_all():
    ###### initialize pages ##########
    session["saveload"] = ""
    app.entitySimilarities: EntitySimilarities = None
    app.scaling: Scaling = None
    app.clustering: Clustering = None
    app.strCostumePlan: list = None
    app.result: list = None
    app.tablelist: list = None
    initialize_costumeplan()
    #app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    ##################################
    return redirect(url_for('home'))

@app.route("/attributes")
def attributes():
    attributes = []
    for attribute in Attribute:
        attribute_string = Attribute.get_name(attribute)
        taxonomieType = Attribute.get_taxonomie_type(attribute)
        if taxonomieType is None:
            tax = "-"
        else:
            tax = TaxonomieType.get_name(taxonomieType)
        attributes.append((attribute_string, tax))
    return render_template("attributes.html", attributes=attributes)

@app.route("/taxonomies")
def taxonomies():
    taxonomies = []
    for taxonomyType in TaxonomieType:
        taxonomyTypeValueString = str(taxonomyType.value)
        taxonomyTypeName = TaxonomieType.get_name(taxonomyType)
        taxonomies.append((taxonomyTypeValueString, taxonomyTypeName))
    return render_template("taxonomies.html", taxonomies=taxonomies)

@app.route("/components")
def components():
    aggregators = []
    for aggregatorType in AggregatorType:
        name = AggregatorType.get_name(aggregatorType)
        description = AggregatorType.get_description(aggregatorType)
        aggregators.append((name, description))

    transformers = []
    for transformerType in TransformerType:
        name = TransformerType.get_name(transformerType)
        description = TransformerType.get_description(transformerType)
        transformers.append((name, description))

    attributeComparers = []
    for attributeComparerType in AttributeComparerType:
        name = AttributeComparerType.get_name(attributeComparerType)
        description = AttributeComparerType.get_description(attributeComparerType)
        attributeComparers.append((name, description))

    elementComparers = []
    for elementComparerType in ElementComparerType:
        name = ElementComparerType.get_name(elementComparerType)
        description = ElementComparerType.get_description(elementComparerType)
        elementComparers.append((name, description))
    
    scalings = []
    for scalingType in ScalingType:
        name = ScalingType.get_name(scalingType)
        description = ScalingType.get_description(scalingType)
        scalings.append((name, description))

    clusterings = []
    for clusteringTyp in ClusteringType:
        name = ClusteringType.get_name(clusteringTyp)
        description = ClusteringType.get_description(clusteringTyp)
        clusterings.append((name, description))
    
    return render_template(
        "components.html", 
        aggregators=aggregators,
        transformers=transformers,
        attributeComparers=attributeComparers,
        elementComparers=elementComparers,
        scalings = scalings,
        clusterings = clusterings
    )

@app.route("/view_taxonomy_<taxonomyValueString>")
def view_taxonomy(taxonomyValueString):
    taxonomyType = TaxonomieType(taxonomyValueString)
    taxonomyName = TaxonomieType.get_name(taxonomyType)
    taxonomy = Taxonomie.create_from_db(taxonomyType)
    taxonomy.save_plot(directory="./static/taxonomies/")

    svg_file_name = "taxonomies/" + taxonomyName + ".svg"

    return "<img src=" + url_for("static", filename=svg_file_name) + " object-fit: contain' >" 

@app.route("/zip_taxonomies")
def zip_taxonomies():
    for taxonomyType in TaxonomieType:
        tax = Taxonomie.create_from_db(taxonomyType)
        tax.save_json("./static/taxonomies/")
        tax.save_plot("./static/taxonomies/")

    base_path = pathlib.Path('./static/taxonomies/')
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as z:
        for f_name in base_path.iterdir():
            z.write(f_name, basename(f_name))
    data.seek(0)
    return send_file(
        data,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename='taxonomies.zip'
    )

# routes for entity plan
@app.route("/costumeplan")
def costumeplan():
    attributes = []
    for attribute in Attribute:
        attributeName = Attribute.get_name(attribute)
        attributeType = attribute
        attributes.append((attributeName, attributeType))

    attributes1, attributes2 = split_list(attributes)


    taxonomies = []
    for taxonomyType in TaxonomieType:
        taxonomyName = TaxonomieType.get_name(taxonomyType)
        taxonomies.append((taxonomyName, taxonomyType))
    
    aggregators = []
    for aggregatorType in AggregatorType:
        aggregatorName = AggregatorType.get_name(aggregatorType)
        aggregators.append((aggregatorName, aggregatorType))

    transformers = []
    for transformerType in TransformerType:
        transformerName = TransformerType.get_name(transformerType)
        transformers.append((transformerName, transformerType))

    attributeComparers = []
    for attributeComparerType in AttributeComparerType:
        attributeComparerTypeName = AttributeComparerType.get_name(attributeComparerType)
        attributeComparers.append((attributeComparerTypeName, attributeComparerType))

    elementComparers = []
    for elementComparerType in ElementComparerType:
        elementComparersName = ElementComparerType.get_name(elementComparerType)
        elementComparers.append((elementComparersName, elementComparerType))

    emptyAttributeActions = []
    for emptyAttributeActionType in EmptyAttributeAction:
        emptyAttributeActionTypeName = EmptyAttributeAction.get_name(emptyAttributeActionType)
        emptyAttributeActions.append((emptyAttributeActionTypeName, emptyAttributeActionType))
    
    return render_template(
        "costumeplan.html",
        emptyAttributeActions = emptyAttributeActions,
        attributes1=attributes1,
        attributes2=attributes2,
        taxonomies=taxonomies,
        aggregators=aggregators,
        transformers=transformers,
        attributeComparers=attributeComparers,
        elementComparers=elementComparers
    )

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

@app.route('/saveload_costume_plan', methods = ['POST', 'GET'])
def saveload_costume_plan():
    if request.method == 'POST':
        COSTUME_PLAN = pickle.loads(session["costumePlan"])
        if request.form['saveload'] == "Save Entity Plan":
            saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.costumePlan)
            saf_cp.set(request.form['session'],COSTUME_PLAN)
            saf_cp.saving()
        elif request.form['saveload'] == "Load Entity Plan":
            saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.costumePlan)
            saf_cp.set(request.form['session'],COSTUME_PLAN)
            test = saf_cp.loading()
            COSTUME_PLAN = test.get_object()
            if isinstance(COSTUME_PLAN, list):
                session["costumePlan"] = pickle.dumps(COSTUME_PLAN)
                session["strCostumePlan"] = costumePlanToStr(COSTUME_PLAN)
            else: 
                flash("For the choosen sessions name ({0}) no plan object exist!".format(request.form['session']))

        session["saveload"] = request.form['session']

    
    return costumeplan()

@app.route("/instance_costume_plan/<value>")
def managing_costume_plan_agre_transf(value):
    value = eval(value)
    costumePlan = pickle.loads(session["costumePlan"])
    if isinstance(value, AggregatorType):
        costumePlan[0] = value
        session["costumePlan"] = pickle.dumps(costumePlan)
        session["strCostumePlan"] = costumePlanToStr(costumePlan)
        
    elif isinstance(value, TransformerType):
        costumePlan[1] = value
        session["costumePlan"] = pickle.dumps(costumePlan)
        session["strCostumePlan"] = costumePlanToStr(costumePlan)
    
    return costumeplan()

def initialize_costumeplan():
    costumePlan = []
    costumePlan.append(AggregatorType.max)
    costumePlan.append(TransformerType.gaussianInverese)
    costumePlan.append((
            Attribute.dominanteFarbe,
            ElementComparerType.wuPalmer,
            AttributeComparerType.singleElement,
            EmptyAttributeAction.ignore
        ))

    session["costumePlan"] = pickle.dumps(costumePlan)
    session["strCostumePlan"] = costumePlanToStr(costumePlan)

@app.route("/reset_costume_plan")
def reset_costume_plan():
    initialize_costumeplan()
    return costumeplan()

@app.route("/instance_costume_plan_attribute_<attribute>")
def managing_costume_plan_set_attribute(attribute: str):
    print(attribute)
    attributetype = eval(attribute)
    costumePlan = pickle.loads(session["costumePlan"])
    costumePlan2 = []
    attributes = []
    for attribute in Attribute:
        attributeName = Attribute.get_name(attribute)
        attributeType = attribute
        attributes.append((attributeName, attributeType))
    

    for plan in costumePlan:
        if (plan == costumePlan[0] or plan == costumePlan[1]):
            costumePlan2.append(plan)

    for attribute in attributes:
        for plan in costumePlan:
            if not (plan == costumePlan[0] or plan == costumePlan[1]):
                if list(plan)[0] == attribute[1] != attributetype:
                    costumePlan2.append(plan)
        if attribute[1] == attributetype:
            check = True
            for plan in costumePlan:
                if not (plan == costumePlan[0] or plan == costumePlan[1]):
                    if list(plan)[0] == attributetype:
                        check = False
            if check:
                costumePlan2.append((attributetype, None , None , None))

    session["costumePlan"] = pickle.dumps(costumePlan2)
    session["strCostumePlan"] = costumePlanToStr(costumePlan2)

    return costumeplan()

@app.route("/instance_costume_plan/<attribute>/<value>")
def managing_costume_plan_attribute(attribute: str , value : str):
    costumePlan = pickle.loads(session["costumePlan"])
    attribute = eval(attribute)
    value = eval(value)
    check = True
    costumePlan2 = costumePlan
    for element in costumePlan:
        if isinstance(element, tuple) and element[0] == attribute:
            if isinstance(value, ElementComparerType):
                index = costumePlan.index(element)
                var = list(element)
                var[1] = value
                element = tuple(var)
                costumePlan2[index] = element
                check = False
            elif isinstance(value, AttributeComparerType):
                index = costumePlan.index(element)
                var = list(element)
                var[2] = value
                element = tuple(var)
                costumePlan2[index] = element
                check = False
            elif isinstance(value, EmptyAttributeAction):
                index = costumePlan.index(element)
                var = list(element)
                var[3] = value
                element = tuple(var)
                costumePlan2[index] = element
                check = False   
    costumePlan = costumePlan2
    if check:
        if isinstance(value, ElementComparerType):
            costumePlan.append((attribute,value,None,None))
        elif isinstance(value, AttributeComparerType):
            costumePlan.append((attribute,None,value,None))
        elif isinstance(value, EmptyAttributeAction):
            costumePlan.append((attribute,None,None,value))
    session["costumePlan"] = pickle.dumps(costumePlan)
    session["strCostumePlan"] = costumePlanToStr(costumePlan)
    return costumeplan()

def costumePlanToStr(costumePlan: list) -> str:

    strCostumePlan = []
    for element in costumePlan:
        if isinstance(element, tuple):
            element = list(element)
            if isinstance(element[0], Attribute):
                element[0] = Attribute.get_name(element[0])
            if isinstance(element[1], ElementComparerType):
                element[1] = ElementComparerType.get_name(element[1])
            else:
                element[1] = None
            if isinstance(element[2], AttributeComparerType):
                element[2] = AttributeComparerType.get_name(element[2])
            else:
                element[2] = None
            if isinstance(element[3], EmptyAttributeAction):
                element[3] = EmptyAttributeAction.get_name(element[3])
            else:
                element[3] = None
            strCostumePlan.append((element[0],element[1],element[2],element[3]))


        elif isinstance(element, AggregatorType):
            strCostumePlan.append(AggregatorType.get_name(element))
        elif isinstance(element, TransformerType):
            strCostumePlan.append(TransformerType.get_name(element))

    return strCostumePlan

@app.route("/load_costume_plan_entitySimilarities")
def load_costume_plan_entitySimilarities():
    if isinstance(app.entitySimilarities, EntitySimilarities):
        session["costumePlan"] = pickle.dumps(app.entitySimilarities.get_costume_plan())
        session["strCostumePlan"] = costumePlanToStr(app.entitySimilarities.get_costume_plan())
    return costumeplan()

# routes for similarities 
@app.route("/entitySimilarities")
def entitySimilarities():
    initialized : bool = False
    memory : bool = False
    showplot: bool = False
    number_costumes =100   #2147483646
    strCostumePlan = ""
    last_sequenz_id = []
    entities_in_memory: str = ""
    min_value = -1
    max_value = -1

    if isinstance(app.entitySimilarities, EntitySimilarities):
        initialized = True
        memory = app.entitySimilarities.get_bool_memory()
        number_costumes = app.entitySimilarities.get_entity_number()
        strCostumePlan =  costumePlanToStr(app.entitySimilarities.get_costume_plan())
        last_sequenz_id = app.entitySimilarities.get_last_sequenz_id()
        entities_in_memory = app.entitySimilarities.get_entities_in_memory()
        if len(last_sequenz_id) != 0:
            min_value = last_sequenz_id[0]
            max_value = last_sequenz_id[-1]
        else:
            min_value = 0
            max_value = number_costumes-1
        
        if len(last_sequenz_id) != 0:
            showplot = True


                
    return render_template(
        "entitySimilarities.html",
        initialized = initialized,
        memory = memory,
        numberCostumes = number_costumes,
        strCostumePlan = strCostumePlan,
        last_sequenz_id = last_sequenz_id,
        min_value = min_value,
        max_value = max_value,
        showplot = showplot,
        EIN = entities_in_memory
    )

@app.route("/entitySimilarities_initialize" , methods = ['POST', 'GET'])
def initialize_entitySimilarities():
    if request.method == 'POST':
        memory = False
        if request.form.get("memory"):
            memory = True
        number_costumes = int(request.form["noic"])
        costumePlan = pickle.loads(session["costumePlan"])
        app.entitySimilarities = EntitySimilarities(costumePlan,memory,number_costumes)
        return entitySimilarities()

@app.route('/saveload_entitySimilarities', methods = ['POST', 'GET'])
def saveload_entitySimilarities():
    if request.method == 'POST':
        if request.form['saveload'] == "Save entitySimilarities":
            saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.entitySimilarities)
            saf_cp.set(request.form['session'],app.entitySimilarities)
            saf_cp.saving()
        elif request.form['saveload'] == "Load entitySimilarities":
            saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.entitySimilarities)
            saf_cp.set(request.form['session'],app.entitySimilarities)
            test = saf_cp.loading()

            if isinstance(test.get_object(), EntitySimilarities):
                app.entitySimilarities = test.get_object()
            else: 
                flash("For the choosen sessions name ({0}) no entitySimilarities object exist!".format(request.form['session']))

            
            
        session["saveload"] = request.form['session']

    
    return entitySimilarities()

@app.route('/entitySimilarities_create_similaritiesMatrix', methods = ['POST', 'GET'])
def entitySimilarities_create_similaritiesMatrix():
    if request.method == 'POST':
        max_value = int(request.form['NoCfSMMax'])
        min_value = int(request.form['NoCfSMMin'])
        similarities = app.entitySimilarities.create_matrix_limited(min_value,max_value)
        sequenz = app.entitySimilarities.get_last_sequenz_id()
        # dfp_instance
        dfp_instance = dfp.DataForPlots(similarities, sequenz ,None,None, None)

        # plot things 
        plt.figure(1)
        G = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(G[0, 0])
        pfc.PlotsForCluster.similarity_plot(dfp_instance, ax1)
        plt.savefig('static/similarities.png', dpi=300, bbox_inches='tight')
        plt.close(1)
    return entitySimilarities()

# scaling page
@app.route('/scaling')
def scaling():
    scalings = []
    for scalingType in ScalingType:
        name = ScalingType.get_name(scalingType)
        scalings.append((name, scalingType))

    exist_scaling: bool = False
    params = []

    if isinstance(app.scaling , Scaling):
        exist_scaling = True
        params = app.scaling.get_param_list()

    return render_template(
        "scaling.html",
        existScaling = exist_scaling,
        scalings = scalings,
        params = params
    )
    
@app.route('/initialize_scaling' , methods = ['POST', 'GET'])
def initialize_scaling():
    if request.method == 'POST':
        scalingType = eval(request.form['scaling'])
        app.scaling = ScalingFactory.create(scalingType)
    return scaling()

@app.route('/set_scaling' , methods = ['POST', 'GET'])
def set_scaling():
    # check for other scaling types the set methodes
    if request.method == 'POST':
        if isinstance(app.scaling, Scaling):
            params = app.scaling.get_param_list()
            params2 = params
            for param in params:
                if param[4] == "number":
                    if param[6] < 1:
                        index = params.index(param)
                        var = list(param)
                        var[3] = float(request.form[param[0]])
                        param = tuple(var)
                        params2[index] = param
                        #print(float(request.form[param[0]]))
                    elif param[6] == 1:
                        index = params.index(param)
                        var = list(param)
                        var[3] = int(request.form[param[0]])
                        param = tuple(var)
                        params2[index] = param
                        #print(int(request.form[param[0]]))
                elif param[4] == "text":
                    if request.form[param[0]] == "inf" or request.form[param[0]] == "np.inf":
                        index = params.index(param)
                        var = list(param)
                        var[3] = np.inf
                        param = tuple(var)
                        params2[index] = param
                        #print(np.inf)
                    elif request.form[param[0]] == "None":
                        index = params.index(param)
                        var = list(param)
                        var[3] = None
                        param = tuple(var)
                        params2[index] = param
                        #print(None)
                    elif isinstance (eval(request.form[param[0]]), float):
                        index = params.index(param)
                        var = list(param)
                        var[3] = float(request.form[param[0]])
                        param = tuple(var)
                        params2[index] = param
                        #print(float(request.form[param[0]]))
                    else:
                        print("no right type found : " + request.form[param[0]])
                elif param[4] == "select":
                    index = params.index(param)
                    var = list(param)
                    var[3] = request.form[param[0]]
                    param = tuple(var)
                    params2[index] = param
                    #print(request.form[param[0]])
                elif param[4] == "checkbox":
                    if request.form.get(param[0]):
                        index = params.index(param)
                        var = list(param)
                        var[3] = True
                        param = tuple(var)
                        params2[index] = param
                        #print(True)
                    else:
                        index = params.index(param)
                        var = list(param)
                        var[3] = False
                        param = tuple(var)
                        params2[index] = param
                        #print(False)

            app.scaling.set_param_list(params2)

    return scaling()

@app.route("/saveload_scaling" , methods = ['POST', 'GET'])
def saveload_scaling():
    if request.method == 'POST':
        if request.form['saveload'] == "Save scaling":
            saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.scaling)
            saf_cp.set(request.form['session'],app.scaling)
            saf_cp.saving()
        elif request.form['saveload'] == "Load scaling":
            saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.scaling)
            saf_cp.set(request.form['session'],app.scaling)
            test = saf_cp.loading()

            if isinstance(test.get_object(), Scaling):
                app.scaling = test.get_object()
            else: 
                flash("For the choosen sessions name ({0}) no scaling object exist!".format(request.form['session']))
            
        session["saveload"] = request.form['session']
    return scaling()

# cluster page 
@app.route('/clustering')
def clustering():
    clusterings = []
    for clusteringTyp in ClusteringType:
        name = ClusteringType.get_name(clusteringTyp)
        clusterings.append((name, clusteringTyp))

    exist_clustering: bool = False
    params = []

    if isinstance(app.clustering , Clustering):
        exist_clustering = True
        params = app.clustering.get_param_list()

    return render_template(
        "clustering.html",
        existClustering = exist_clustering,
        clusterings = clusterings,
        params = params
    )
    
@app.route('/initialize_clustering' , methods = ['POST', 'GET'])
def initialize_clustering():
    if request.method == 'POST':
        clusteringType = eval(request.form['clustering'])
        app.clustering = ClusteringFactory.create(clusteringType)
    return clustering()

@app.route('/set_clustering' , methods = ['POST', 'GET'])
def set_clustering():
    # check for other scaling types the set methodes
    if request.method == 'POST':
        if isinstance(app.clustering, Clustering):
            params = app.clustering.get_param_list()
            params2 = params
            for param in params:
                if param[4] == "number":
                    if param[6] < 1:
                        index = params.index(param)
                        var = list(param)
                        var[3] = float(request.form[param[0]])
                        param = tuple(var)
                        params2[index] = param
                        #print(float(request.form[param[0]]))
                    elif param[6] == 1:
                        index = params.index(param)
                        var = list(param)
                        var[3] = int(request.form[param[0]])
                        param = tuple(var)
                        params2[index] = param
                        #print(int(request.form[param[0]]))
                elif param[4] == "text":
                    if request.form[param[0]] == "inf" or request.form[param[0]] == "np.inf":
                        index = params.index(param)
                        var = list(param)
                        var[3] = np.inf
                        param = tuple(var)
                        params2[index] = param
                        #print(np.inf)
                    elif request.form[param[0]] == "None":
                        index = params.index(param)
                        var = list(param)
                        var[3] = None
                        param = tuple(var)
                        params2[index] = param
                        #print(None)
                    elif isinstance (eval(request.form[param[0]]), float):
                        index = params.index(param)
                        var = list(param)
                        var[3] = float(request.form[param[0]])
                        param = tuple(var)
                        params2[index] = param
                        #print(float(request.form[param[0]]))
                    else:
                        print("no right type found : " + request.form[param[0]])
                elif param[4] == "select":
                    index = params.index(param)
                    var = list(param)
                    var[3] = request.form[param[0]]
                    param = tuple(var)
                    params2[index] = param
                    #print(request.form[param[0]])
                elif param[4] == "checkbox":
                    if request.form.get(param[0]):
                        index = params.index(param)
                        var = list(param)
                        var[3] = True
                        param = tuple(var)
                        params2[index] = param
                        #print(True)
                    else:
                        index = params.index(param)
                        var = list(param)
                        var[3] = False
                        param = tuple(var)
                        params2[index] = param
                        #print(False)

            app.clustering.set_param_list(params2)

    return clustering()

@app.route("/saveload_clustering" , methods = ['POST', 'GET'])
def saveload_clustering():
    if request.method == 'POST':
        if request.form['saveload'] == "Save clustering":
            saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.clustering)
            saf_cp.set(request.form['session'],app.clustering)
            saf_cp.saving()
        elif request.form['saveload'] == "Load clustering":
            saf_cp = sal.SavingAndLoadingFactory.create(sal.SavingAndLoadingType.clustering)
            saf_cp.set(request.form['session'],app.clustering)
            test = saf_cp.loading()

            if isinstance(test.get_object(), Clustering):
                app.clustering = test.get_object()
            else: 
                flash("For the choosen sessions name ({0}) no clustering object exist!".format(request.form['session']))

        session["saveload"] = request.form['session']
    return clustering()

# assume and calculating
@app.route("/calculating")
def calculating():
    #app.strCostumePlan = session["strCostumePlan"]
    
    simiParams: list = []
    if isinstance(app.entitySimilarities , EntitySimilarities):
        simiParams = app.entitySimilarities.get_param_list()
        costume_plan = costumePlanToStr(app.entitySimilarities.get_costume_plan())
    else:
        simiParams.append(("errorentitySimilarities", "no Entity Similarities Type initialized", " go to entitySimilarities and initialize","failed"))
        costume_plan = ["No" , "Costume" ,("are", "initialized!","Initialize" ,"entitySimilarity!" )]

    scalingParams: list = []
    if isinstance(app.scaling , Scaling):
        scalingParams = app.scaling.get_param_list()
    else:
        scalingParams.append(("errorScaling", "no Scaling Type initialized", " go to scaling and initialize","failed"))

    clusteringParams: list = []
    if isinstance(app.clustering , Clustering):
        clusteringParams = app.clustering.get_param_list()
    else:
        clusteringParams.append(("errorClustering", "no Clustering Type initialized", " go to clustering and initialize","failed"))

    #flash('Thank you for registering')
    return render_template(
        "calculating.html",
        strCostumePlan = costume_plan,
        simiParams = simiParams,
        scalingParams = scalingParams,
        clusteringParams = clusteringParams
    )

# start calculating
@app.route("/start_calculating" , methods = ['POST', 'GET'] )
def start_calculating():    
    params = []

    try:
        min_value = int(request.form['min'])
        max_value = int(request.form['max'])
        params.append(("minEntity" , "Min. Entity" , "description" , min_value , "header"))
        params.append(("maxEntity" , "Max. Entity" , "description" , max_value , "header"))
        
    except Exception as error:
        flash(" an Error occurs in check min max value. Please try again. Error: " + str(error))
        return calculating()

    similarities: np.matrix
    try:
        similarities = app.entitySimilarities.create_matrix_limited(min_value,max_value)
        sequenz = app.entitySimilarities.get_last_sequenz_id()
        params.append(("similarityMatrix" , "Similarity Matrix" , "description" , similarities , "header"))
        params.append(("lastSequenzID" , "Last Sequenz ID" , "description" , sequenz , "header"))
    except Exception as error:
        flash(" an Error occurs in creating similarity matrix. Please try again. Error: " + str(error))
        return calculating()
    
    try:
        pos: np.matrix 
        pos = app.scaling.scaling(similarities)
        stress = app.scaling.stress_level()
        
        pos2d: np.matrix 
        dim: int = app.scaling.get_dimensions()
        app.scaling.set_dimensions(2)
        pos2d = app.scaling.scaling(similarities)
        app.scaling.set_dimensions(dim)
        params.append(("positionMatrixND" , "Position Matrix n-Dimensional" , "description" , pos , "header"))
        params.append(("positionMatrix2D" , "Position Matrix 2-Dimensional" , "description" , pos2d , "header"))
        params.append(("stressLevel" , "Stress Level" , "description" , stress , "header"))
    except Exception as error:
        flash(" an Error occurs in creating position matrix. Please try again. Error: " + str(error))
        return calculating()

    try:
        labels: np.matrix
        labels = app.clustering.create_cluster(pos)
        params.append(("labels" , "Label Matrix" , "description" , labels , "header"))
    except Exception as error:
        flash(" an Error occurs in creating labels. Please try again. Error: " + str(error))
        return calculating()


    # dfp_instance
    try:
        dfp_instance = dfp.DataForPlots(similarities, sequenz ,None,pos2d, labels)
    except Exception as error:
        flash(" an Error occurs in creating Data for Plot Instance. Please try again. Error: " + str(error))
        return calculating()

    try:
        plt.figure(1)
        G = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(G[0, 0])
        pfc.PlotsForCluster.similarity_plot(dfp_instance, ax1)
        plt.savefig('static/similarities.png', dpi=300, bbox_inches='tight')
        plt.close(1)

        plt.figure(1)
        G = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(G[0, 0])
        pfc.PlotsForCluster.scaling_2d_plot(dfp_instance, ax1)
        plt.savefig('static/scaling.png', dpi=300, bbox_inches='tight')
        plt.close(1)

        plt.figure(1)
        G = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(G[0, 0])
        pfc.PlotsForCluster.cluster_2d_plot(dfp_instance ,ax1)
        plt.savefig('static/clustering.png', dpi=300, bbox_inches='tight')
        plt.close(1)

    except Exception as error:
        flash(" an Error occurs in creating Plots. Please try again. Error: " + str(error))
        return calculating()

    app.result = params

    return redirect(url_for('result'))

# methodes for result 
@app.route("/result")
def result():
    if isinstance(app.result , list):
        return render_template(
            "result.html",
            params = app.result
        )
    else:
        flash('No Results are calculated! Redirection to calculation!')
        return calculating()

@app.route("/view_result_<value>")
def view_result(value):

    if value == "similarityMatrix":
        return "<img src=" + url_for("static", filename="similarities.png") + " object-fit: contain' >" 

    if value == "positionMatrix2D":
        return "<img src=" + url_for("static", filename="scaling.png") + " object-fit: contain' >"
    
    if value == "labels":
        return "<img src=" + url_for("static", filename="clustering.png") + " object-fit: contain' >"

@app.route("/result_value_similarity", methods = ['POST', 'GET'])
def result_value_similarity():
    try:
        if request.method == 'POST':
            min_value = int(request.form['min'])
            max_value = int(request.form['max'])
            for param in app.result:
                if list(param)[0] == "minEntity":
                    lowest_value = list(param)[3]
            for param in app.result:
                if list(param)[0] == "maxEntity":
                    highest_value = list(param)[3]
            if min_value >= lowest_value and min_value <= highest_value:
                if max_value >= lowest_value and max_value <= highest_value:
                    for param in app.result:
                        if list(param)[0] == "similarityMatrix":
                            matrix = list(param)[3]
                            flash(" similarity value: {0} to {1} = {2} \n similarity value: {1} to {0} = {3}".format(min_value,max_value, round(matrix[min_value-lowest_value][max_value-lowest_value],2),round(matrix[max_value-lowest_value][min_value-lowest_value],2)), "similarity")
                            return result()
        flash("Error please check the input!", "similarity")
    except Exception:   
        flash("Error please check the input!", "similarity")
    
    return result()            
            
@app.route("/result_coordinates", methods = ['POST', 'GET'])
def result_coordinates():
    try:
        if request.method == 'POST':
            value = int(request.form['coordinates'])
            for param in app.result:
                if list(param)[0] == "minEntity":
                    lowest_value = list(param)[3]
            for param in app.result:
                if list(param)[0] == "maxEntity":
                    highest_value = list(param)[3]
            if value >= lowest_value and value <= highest_value:
                    for param in app.result:
                        if list(param)[0] == "positionMatrixND":
                            matrix = list(param)[3]
                            flash("Coordinates of Entity {0}: {1}".format(value,matrix[value-highest_value][:]), "scaling")
                            return result()
        flash("Error please check the input!" , "scaling")
    except Exception:   
        flash("Error please check the input!" , "scaling")
    
    return result()
        
@app.route("/result_cluster", methods = ['POST', 'GET'])
def result_cluster():
    try:
        id_cluster: list = []
        if request.method == 'POST':
            value = int(request.form['clusterEntity'])
            for param in app.result:
                if list(param)[0] == "lastSequenzID":
                    sequenz_id = list(param)[3]
            for param in app.result:
                if list(param)[0] == "labels":
                    labels = list(param)[3]
                    highest_value = np.amax(labels)
                    if value >= -1 and value <= highest_value:
                        for i in range(len(labels)):
                            if labels[i] == value:
                                id_cluster.append(sequenz_id[i])
                        flash("Entities ID for Cluster {0}: {1}".format(value,id_cluster) , "cluster")
                        return result()
        flash("Error please check the input!" , "cluster")
    except Exception:   
        flash("Error please check the input!" , "cluster")
    
    return result()
    
@app.route("/table_list")
def table_list():
    tablelist_bool: bool = True
    entitySimi_bool: bool = True
    if not isinstance(app.tablelist , list):
        tablelist_bool = False
        flash("No table list generated")
    if not isinstance(app.entitySimilarities , EntitySimilarities):
        entitySimi_bool = False
        flash("No EntitySimilarities Object instantiated")

    return render_template(
            "entityTable.html",
            table = app.tablelist,
            tableListBool = tablelist_bool,
            entitySimiBool = entitySimi_bool
        )

@app.route("/instance_table_costume_plan" , methods = ['POST', 'GET'])
def instance_table_costume_plan():

    sequenz_id = eval(request.form['sequenzId'])
    
    if not isinstance(app.entitySimilarities , EntitySimilarities):
        return home()

    
    
    entities = app.entitySimilarities.get_list_entities()
    costumeplan = app.entitySimilarities.get_costume_plan()
    show_table_list = []
    header_list = []
    attributes_list = []
    
    header_list.append("ID")
    header_list.append("Referenz Film")
    header_list.append("Referenz Rollen")
    header_list.append("Referenz Kostuem")
    
    for plan in costumeplan:
        if not (plan == costumeplan[0] or plan == costumeplan[1]):
            attributes_list.append(list(plan)[0])
            header_list.append(Attribute.get_name(list(plan)[0]))
    
    show_table_list.append(tuple(header_list))
    
    for entity in entities:
        if entity.id in sequenz_id:
            entityList = []
            entityList.append(entity.id)
            entityList.append(entity.get_film_url())
            entityList.append(entity.get_rollen_url())
            entityList.append(entity.get_kostuem_url())
            for attribute in attributes_list:
                entityList.append(entity.values[attribute])
            show_table_list.append(tuple(entityList))
    app.tablelist = show_table_list

    return redirect(url_for('table_list'))

if __name__ == '__main__':
    port = 5001
    url = "http://127.0.0.1:{0}".format(port)
    
    # prevent opening 2 tabs if debug=True
    if "WERKZEUG_RUN_MAIN" not in os.environ:
        # starts opening the tab if server is up
        threading.Timer(1.2, lambda: webbrowser.open(url) ).start()

    Logger.normal("Instance directory is " + app.instance_path)
    app.run(port=port, debug=True)
