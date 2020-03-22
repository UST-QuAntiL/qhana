import tkinter as tk
import pygubu
import sys
#sys.path.append('..\backend')
from backend.costume import Costume
import backend.elementComparer as elemcomp
import backend.attributeComparer as attrcomp
import backend.aggregator as aggre
from backend.costumeComparer import CostumeComparer
from backend.database import Database
from backend.taxonomie import Taxonomie
from backend.attribute import Attribute
import re
from PIL import Image, ImageTk, PngImagePlugin
from cairosvg import svg2png


class Application:
    def __init__(self, master):
        self.master = master

        #Create builder
        self.builder = builder = pygubu.Builder()
        # Load ui file 
        builder.add_from_file('muse.ui')
        #Create a widget, using master as parent
        self.mainwindow = builder.get_object('mainwindow', master)

        #Connect callbacks
        builder.connect_callbacks(self)

    def start_button_click(self):
        #connect to databank 
        active_label = self.builder.get_object('active_label')
        try:
            active_label.configure(text='load from database')
            tax = Taxonomie()
            tax.load_all()    
            active_label.configure(text='connection was successfull')

        except:
            active_label.configure(text='connection fails ')
        
        taxobox_value = self.builder.get_object('taxobox').get()
        path = path = "plots/" + taxobox_value +".svg"
        print(path)
        active_showbox = self.builder.get_object('svg_shower')
        item = svg2png(url=path, parent_width = "50", parent_height = "50")
        self.tkimage = ImageTk.PhotoImage(data=item)
        active_showbox.configure(image = self.tkimage)
        # traits invalid value (too big) for the size of the input
        

    
    #quit by click on quit 
    def on_quit_button_click(self):
        """Quit button callback"""
        self.master.quit()
    
    

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)

    root.mainloop()
