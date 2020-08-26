import numpy as np
import os
import pandas as pd
from keras.models import load_model
#from tensorflow.python.keras.backend import set_session
#from tensorflow.python.keras.models import load_model

from keras.preprocessing import image
import tensorflow as tf
global graph
#global sess
#tf_config = some_custom_config
#sess=tf.Session(config=tf_config)
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
#set_session(sess)

@app.route('/')
def index():
    return render_template('pn1.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (96,96))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            model = load_model("Model_1.h5")
            models=[model]
            for i,model in enumerate(models):
                preds = model.predict_classes(x)
            #set_session(sess)
            #preds = model.predict_classes(x)
            
            print("prediction",preds)
            
         
        index = ['NORMAL','PNEUMONIA']
        
        text = "the result is : " + str(index[preds[0][0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    
