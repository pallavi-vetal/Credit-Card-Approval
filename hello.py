from cloudant import Cloudant
import atexit
import os
import json
from flask import Flask, flash , redirect, render_template , request, session, abort , Markup
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from werkzeug import secure_filename
import json
from scipy.stats import kurtosis, skew
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from werkzeug import secure_filename

app = Flask(__name__)
app = Flask(__name__)
app.secret_key = os.urandom(12)

dropdown_list = []
dropdown_list_2 = []

db_name = 'mydb'
client = None
db = None

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif "CLOUDANT_URL" in os.environ:
    client = Cloudant(os.environ['CLOUDANT_USERNAME'], os.environ['CLOUDANT_PASSWORD'], url=os.environ['CLOUDANT_URL'], connect=True)
    db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)


port = int(os.getenv('PORT', 8000))




# In[3]:


UPLOAD_FOLDER = './Uploads'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# In[4]:


@app.route('/upload')
def upload_file():
    dropdown_list.clear()
    dropdown_list_2.clear()
    return render_template('upload.html')


# In[5]:


@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
     if request.method == 'POST':
        
        if 'file' not in request.files:
            flash('No file part')
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file.filename)
        if file.filename == '':
            flash('No selected file')
            print('No selected file')
            return redirect(request.url)
        if file :
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploadFileAndPredict(fpath)
            with open(fpath) as file:
                allRead = csv.reader(file, delimiter=',')
                lineCount = 0
                for row in allRead:
                    if lineCount==0:
                        lineCount=lineCount+1
                    else:
                        lineCount=lineCount+1
                        dropdown_list_2.append((row[0]))
            return render_template('Result.html',  dropdown_list_2=dropdown_list_2)


# In[6]:


@app.route('/input_percent' , methods = ['GET','POST'])
def input_num():
    x = request.form["in"]
    fpath = os.path.join("default", "output.csv")
    line = pd.read_csv(fpath).shape[0]
    y = round((float(x)*line)/100)
    print(line)
    print(y)
    ls = []
    lschurn=[]
    with open(fpath) as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount == 0:
                lineCount += 1
            elif lineCount <= y and lineCount != 0:
                ls.append(row[0])
                lschurn.append(row[16])
                lineCount += 1
    lss=list(map(lambda x: float(x*100),list(pd.read_csv(fpath)['Approved'][:y].copy())))  
    print(lss)
    print(lschurn)
    return render_template('Percent.html', outList = ls, value_list=lschurn,values_res=lss )


# In[7]:


@app.route('/check/<string:dropdown>',methods=['POST','GET'])
def specific(dropdown):
    x = dropdown
    x = search_default(x)
    #Key,Male,Age,Debt,Married,BankCustomer,EducationLevel,Ethnicity,YearsEmployed,PriorDefault,Employed,
    #CreditScore,DriversLicense,Citizen,ZipCode,Income,Approved

    key  = x[0]
    Male = x[1]
    Age = x[2]
    Married  = x[3]
    Debt = x[4]
    BankCustomer  = x[5]
    EducationLevel  = x[6]
    Ethnicity = x[7]
    YearsEmployed  = x[8]
    PriorDefault = x[9]
    Employed = x[10]
    CreditScore = x[11]
    Income = x[15]    
    x = x[16]
    pred= float(x)*100
    values = [pred]
    x = float(x)*100
    x = round(x,2)
    
    return render_template('Chart.html', key=key, 
                           Male=Male, Age=Age, Married=Married, Debt=Debt, 
                           BankCustomer=BankCustomer, EducationLevel=EducationLevel, Ethnicity=Ethnicity,
                           YearsEmployed=YearsEmployed, PriorDefault=PriorDefault, 
                           Employed=Employed, CreditScore = CreditScore,
                           Income = Income, values = values,pred=x)


# In[8]:


def preprocess_data(fileInput):
    train = pd.read_csv(fileInput)  
    train.replace('?', np.NaN, inplace = True)
    train['YearsEmployed'] = [x*100 for x in train['YearsEmployed']]
    train['Age'] = train['Age'].astype('float64')
    train['Age'].fillna((train['Age'].mean()), inplace=True)
    train['Married'] = train['Married'].astype('category')
    train['BankCustomer'] = train['BankCustomer'].astype('category')
    train['EducationLevel'] = train['EducationLevel'].astype('category')
    train['Ethnicity'] = train['Ethnicity'].astype('category')
    train['PriorDefault'] = train['PriorDefault'].astype('category')
    train['Employed'] = train['Employed'].astype('category')
    train['DriversLicense'] = train['DriversLicense'].astype('category')
    train['Citizen'] = train['Citizen'].astype('category')

    cat_columns = train.select_dtypes(['category']).columns
    train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)
    train = train.drop(['ZipCode','Male'], axis=1)
    X_test = train.iloc[:,1:].values

    return X_test
    
    


# In[9]:


@app.route('/')
def home():
        return render_template('upload.html')


# In[10]:


def uploadFileAndPredict(filename):
    K.clear_session()
    dropdown_list_2.clear()
    proceseed_data = preprocess_data(filename)
    model = pickle.load(open('LRModel.pkl', 'rb'))
    y_pred = model.predict_proba(proceseed_data)
    df = pd.read_csv(filename)
    df['Approved'] = [x[0] for x in y_pred]
    df.set_index('Key', inplace=True)
    df.sort_values('Approved', ascending=False, inplace=True)
    fpathr = os.path.join("default", "output.csv")
    df.to_csv(fpathr)  
    return y_pred
             
    


# In[11]:


@app.route('/defaultfile', methods = ['GET', 'POST'])
def uploader_default_file():
    fpath = os.path.join("default", "test.csv")
    uploadFileAndPredict(fpath)
    with open(fpath) as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount==0:
                lineCount=lineCount+1
            else:
                lineCount=lineCount+1
                dropdown_list_2.append((row[0]))
    return render_template('Result.html',  dropdown_list_2=dropdown_list_2)


# In[12]:


@app.route('/check_default/<string:dropdown_2>',methods=['POST','GET'])
def specific_default(dropdown_2):
    x = dropdown_2
    
    x = search_default(x)
    key  = x[0]
    Male = x[1]
    Age = x[2]
    Married  = x[3]
    Debt = x[4]
    BankCustomer  = x[5]
    EducationLevel  = x[6]
    Ethnicity = x[7]
    YearsEmployed  = x[8]
    PriorDefault = x[9]
    Employed = x[10]
    CreditScore = x[11]
    Income = x[15]    
    x = x[16]
    pred= float(x)*100
    values = [pred]
    x = float(x)*100
    x = round(x,2)
    
    return render_template('Chart.html', key=key, 
                           Male=Male, Age=Age, Married=Married, Debt=Debt, 
                           BankCustomer=BankCustomer, EducationLevel=EducationLevel, Ethnicity=Ethnicity,
                           YearsEmployed=YearsEmployed, PriorDefault=PriorDefault, 
                           Employed=Employed, CreditScore = CreditScore,
                           Income = Income, values = values,pred=x)


# In[13]:


def search_default(cid):
    fpathr = os.path.join("default", "output.csv")
    with open(fpathr) as file:
        allRead = csv.reader(file, delimiter=',')
        for row in allRead:
            if row[0]==cid:
                return row

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
