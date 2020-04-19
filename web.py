from flask import Flask, render_template
import threading, webbrowser
from backend.attribute import Attribute

app = Flask(__name__)

@app.route('/')
def home():
    attributes = []
    for attribute in Attribute:
        attributes.append(Attribute.get_name(attribute))
    return render_template("home.html", attributes=attributes)

if __name__ == '__main__':
    port = 5001
    url = "http://127.0.0.1:{0}".format(port)
    threading.Timer(0.0, lambda: webbrowser.open(url) ).start()
    app.run(port=port, debug=True)
