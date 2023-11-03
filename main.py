import numpy as np
import os
from kivy.app import App

from kivy.lang import Builder

from kivy.uix.image import Image as Im
from kivymd.app import MDApp
from PIL import Image
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.factory import Factory
from kivymd.uix.banner import MDBanner
from kivy.uix.button import Button
from model import TensorFlowModel
from kivy.uix.boxlayout import BoxLayout
import os
import sys

CDC = 'class'
prediction = 'confidence'
current_directory = os.getcwd()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2   
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path) 


class cameraApp(MDApp):

    def build(self):
        global input_img
        
        layout = BoxLayout(orientation='vertical')

        # create camera instance
        self.cam = Camera(resolution = (640, 480))

        # create button
        self.btn = MDRectangleFlatButton(
            
            text="Capture",
            pos_hint={"center_x": .5, "center_y": .153},
            size_hint=(1, 0.2),
            padding = "12dp",
            width = "200dp",
            on_press=self.capture_image)
        
        self.btn2 = MDRectangleFlatButton(
            
            text="Upload",
            pos_hint={"center_x": .5, "center_y": .2},
            size_hint=(1, 0.2),
            on_press=self.file_manager_open)
        
 #       self.runButton = MDRoundFlatIconButton(text="Run", icon="check",disabled=True, pos_hint={"center_x": .5, "center_y": .53})
        
        

        # create label
        self.lbl_class = MDLabel(
            text=CDC,
            size_hint=(1, 0.2))

        self.lbl_conf = MDLabel(
            text=prediction,
            size_hint=(1, 0.2))
        
        self.box_lay=MDBoxLayout(adaptive_height= True)
        
        
        self.disp_img=Im(
        size_hint_x=0.30,
        pos_hint = {"center_x":0.5, "center_y":0.1})
        
        #self.theme_cls.colors = colors
        #self.theme_cls.primary_palette = "Blue"
        self.theme_cls.primary_palette = "Green"
        self.result = MDTopAppBar(title="Result")
        
        # add widgets in layout
        layout.add_widget(self.cam)
        layout.add_widget(self.btn)
        layout.add_widget(self.btn2)
        layout.add_widget(self.disp_img)
        #layout.add_widget(self.box_lay)
        layout.add_widget(self.result)
        layout.add_widget(self.lbl_class)
        layout.add_widget(self.lbl_conf)
        
        return layout

    def capture_image(self, *args):
        # save captured image

        self.cam.export_to_png(os.path.join(os.getcwd(), 'img.jpg'))
        input_img='img.png'
        butterfly, prediction = self.predict()
        self.lbl_class.text = butterfly
        self.lbl_conf.text = prediction
        self.disp_img.source = 'img.jpg'
#        self.disp_img.source = Image.open('img.png')
    
    def file_manager_open(self,path):
        
        from plyer import filechooser
        path = filechooser.open_file(filters=["*.jpg"],on_selection=self.select_path)[0]
    
    def select_path(self, path): 
        global uploaded
        if path:
            #self.disp_img.export_to_png(path[0], 'img.png')
            self.disp_img.source=path[0]
            picture=Image.open(path[0])
            picture = picture.save("img.jpg")
            butterfly, prediction = self.predict()
            self.lbl_class.text = butterfly
            self.lbl_conf.text = prediction    
            
    def predict(self, model='VGG16.tflite'):
        butterflies = [
            
            'Daun Sehat', 
            'Daun Sakit CDC Ringan', 
            'Daun Sakit CDC Berat']

        # Load TFLite model
        model_to_pred = TensorFlowModel()
        model_to_pred.load(os.path.join(current_directory, model))

        # Read image and predict
        img = Image.open(os.path.join(os.getcwd(), 'img.jpg'))
        img_arr = np.array(img.resize((224, 224)), np.float32)
        img_arr = img_arr[:, :, :3]
        img_arr = np.expand_dims(img_arr, axis=0)
        preds = dict(zip(butterflies, list(model_to_pred.pred(img_arr)[0])))
        best = max(preds, key=preds.get)

        return best, str(preds[best]*100)

if __name__ == '__main__':
    #run app
    cameraApp().run()

