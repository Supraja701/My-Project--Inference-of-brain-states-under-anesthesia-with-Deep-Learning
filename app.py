from flask import *
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

filename = 'pickle.pkl'
classifier = pickle.load(open(filename, 'rb'))

@app.route("/")
def home():
    return render_template("browser1.html")  
@app.route('/login',methods = ['POST'])  
def login():  
      fil=request.form['files']
      store=pd.read_csv(fil)
      le=LabelEncoder()
      type(store)
      op=classifier.predict(store)
      if op[0]==0:
          return render_template('index1.html')
      elif op[0]==1:
          return render_template('index2.html')
      elif op[0]==2:
          return render_template('index3.html')

if __name__ == '__main__':
   app.run()  
##
