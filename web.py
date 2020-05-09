from flask import Flask, render_template, url_for, session, send_file
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

app = Flask(__name__)
app.secret_key = "super secret key"

@app.route("/")
def home():
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
    
    return render_template(
        "components.html", 
        aggregators=aggregators,
        transformers=transformers,
        attributeComparers=attributeComparers,
        elementComparers=elementComparers
    )


@app.route("/view_taxonomy_<taxonomyValueString>")
def view_taxonomy(taxonomyValueString):
    db = Database()
    db.open()

    taxonomyType = TaxonomieType(taxonomyValueString)
    taxonomyName = TaxonomieType.get_name(taxonomyType)
    taxonomy = Taxonomie.create_from_db(taxonomyType, db)
    taxonomy.save_plot(directory="static")

    svg_file_name = taxonomyName + ".svg"

    return "<img src=" + url_for("static", filename=svg_file_name) + " object-fit: contain' >" 

@app.route("/zip_taxonomies")
def zip_taxonomies():
    db = Database()
    db.open()

    for taxonomyType in TaxonomieType:
        tax = Taxonomie.create_from_db(taxonomyType, db)
        tax.save_json("taxonomies")
        tax.save_plot("taxonomies")

    base_path = pathlib.Path('./taxonomies/')
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
@app.route("/costumeplan")
def costumeplan():
    attributes = []
    for attribute in Attribute:
        attributeTypeValueString = str(Attribute.get_name(attribute))
        attributeTypeName = Attribute.get_name(attribute)
        attributes.append((attributeTypeValueString, attributeTypeName))

    taxonomies = []
    for taxonomyType in TaxonomieType:
        taxonomyTypeValueString = str(taxonomyType.value)
        taxonomyTypeName = TaxonomieType.get_name(taxonomyType)
        taxonomies.append((taxonomyTypeValueString, taxonomyTypeName))
    
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

    emptyAttributeActions = []
    for emptyAttributeActionType in EmptyAttributeAction:
        name = EmptyAttributeAction.get_name(emptyAttributeActionType)
        action_type = EmptyAttributeAction.get_emptyAttributeAction_type(emptyAttributeActionType)
        emptyAttributeActions.append((name, action_type))

    costumeplan = [
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
    

    
    
    return render_template(
        "costumeplan.html",
        costumeplan = str(costumeplan),
        emptyAttributeActions = emptyAttributeActions,
        attributes=attributes,
        taxonomies=taxonomies,
        aggregators=aggregators,
        transformers=transformers,
        attributeComparers=attributeComparers,
        elementComparers=elementComparers
    )

if __name__ == '__main__':
    port = 5001
    url = "http://127.0.0.1:{0}".format(port)
    
    # prevent opening 2 tabs if debug=True
    if "WERKZEUG_RUN_MAIN" not in os.environ:
        # starts opening the tab if server is up
        threading.Timer(1.2, lambda: webbrowser.open(url) ).start()

    Logger.normal("Instance directory is " + app.instance_path)
    app.run(port=port, debug=True)
