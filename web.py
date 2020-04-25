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

if __name__ == '__main__':
    port = 5001
    url = "http://127.0.0.1:{0}".format(port)
    
    # prevent opening 2 tabs if debug=True
    if "WERKZEUG_RUN_MAIN" not in os.environ:
        # starts opening the tab if server is up
        threading.Timer(1.2, lambda: webbrowser.open(url) ).start()

    Logger.normal("Instance directory is " + app.instance_path)
    app.run(port=port, debug=True)
