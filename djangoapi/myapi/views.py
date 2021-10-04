from django.shortcuts import render
from rest_framework import viewsets
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from . forms import ApprovalForm
from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib import messages
from . models import approvals
from . serializers import approvalsSerializers

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import warnings
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
import pickle
import joblib

class ApprovalsView(viewsets.ModelViewSet):
    queryset = approvals.objects.all()
    serializer_class = approvalsSerializers

def myform(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            myform = form.save(commit = False)
        else:
            form = Myform()

def ohevalue(df):
    file = open("./myapi/pklfiles/allcol.pkl",'rb')
    ohe_col = pickle.load(file)
    file.close()
    cat_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    df_processed = pd.get_dummies(df, columns=cat_columns)
    newdict = {}
    for i in ohe_col:
        if i in df_processed.columns:
            newdict[i] = df_processed[i].values
        else:
            newdict[i] = 0
    newdf = pd.DataFrame(newdict)
    return newdf

def approvereject(test):
    try:
        test['LoanAmount'] = test['LoanAmount'].astype('int')
        test = test.reset_index(drop=True)
        
        encoder = pickle.load(open("./myapi/pklfiles/ohe_rev00.sav", 'rb'))
        encoded_data = encoder.transform(test[['Gender','Married','Education','Self_Employed','Property_Area']])
        encoded_df = pd.DataFrame(encoded_data, columns = ['Gender_Female', 'Gender_Male', \
        'Married_No', 'Married_Yes', \
        'Education_Graduate', 'Education_Not Graduate', \
        'Self_Employed_No', 'Self_Employed_Yes', \
        'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'])
        
        test_x = pd.concat([test[['Dependents','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']], 
                    encoded_df], axis = 1)
        
        scaler_filename = "./myapi/pklfiles/scaler_rev00.sav"
        scaler = joblib.load(scaler_filename)
        test_x = scaler.fit_transform(test_x)
        
        filename = './myapi/pklfiles/svm_rev00.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(test_x)
        
        if y_pred[0] == 1:
            print('Approved')
            return 'Approved'
        elif y_pred[0] == 0:
            print('Rejected')
            return 'Rejected'
        
    except ValueError as e:
        return (e.args[0])
        

def cxcontact(request):
    if request.method=='POST':
        form=ApprovalForm(request.POST)
        if form.is_valid():
                Firstname = form.cleaned_data['firstname']
                Lastname = form.cleaned_data['lastname']
                Dependents = form.cleaned_data['Dependents']
                ApplicantIncome = form.cleaned_data['ApplicantIncome']
                CoapplicantIncome = form.cleaned_data['CoapplicantIncome']
                LoanAmount = form.cleaned_data['LoanAmount']
                Loan_Amount_Term = form.cleaned_data['Loan_Amount_Term']
                Credit_History = form.cleaned_data['Credit_History']
                Gender = form.cleaned_data['Gender']
                Married = form.cleaned_data['Married']
                Education = form.cleaned_data['Education']
                Self_Employed = form.cleaned_data['Self_Employed']
                Property_Area = form.cleaned_data['Property_Area']
                myDict = (request.POST).dict()
                df=pd.DataFrame(myDict, index=[0])
                
                print('######' , df)
                
                answer=approvereject(df)
                
                if int(df['LoanAmount'])<25000:
                    messages.success(request,'Application Status: {}'.format(answer))
                else:
                    messages.success(request,'Invalid: Your Loan Request Exceeds the $25,000 Limit')

    form=ApprovalForm()

    return render(request, 'myform/cxform.html', {'form':form})

def cxcontact2(request):
    if request.method=='POST':
        form=ApprovalForm(request.POST)
        if form.is_valid():
                Firstname = form.cleaned_data['firstname']
                Lastname = form.cleaned_data['lastname']
                Dependents = form.cleaned_data['Dependents']
                ApplicantIncome = form.cleaned_data['ApplicantIncome']
                CoapplicantIncome = form.cleaned_data['CoapplicantIncome']
                LoanAmount = form.cleaned_data['LoanAmount']
                Loan_Amount_Term = form.cleaned_data['Loan_Amount_Term']
                Credit_History = form.cleaned_data['Credit_History']
                Gender = form.cleaned_data['Gender']
                Married = form.cleaned_data['Married']
                Education = form.cleaned_data['Education']
                Self_Employed = form.cleaned_data['Self_Employed']
                Property_Area = form.cleaned_data['Property_Area']
                myDict = (request.POST).dict()
                df=pd.DataFrame(myDict, index=[0])
                answer=approvereject(ohevalue(df))[0]
                Xscalers=approvereject(ohevalue(df))[1]
                messages.success(request,'Application Status: {}'.format(answer))

    form=ApprovalForm()

    return render(request, 'myform/cxform.html', {'form':form})
    
    
    
'''
@api_view(["POST"])
def approvereject(request):
    try:
        mdl = tensorflow.keras.model.load_model("/pklfiles/rev00/")
        #mydata=pd.read_excel('/pklfiles/bankloan.csv')
        mydata=request.data
        unit=np.array(list(mydata.values()))
        test=unit.reshape(1,-1)

        print(type(test))
        print(test)

        scalers=joblib.load("/pklfiles/scalers.pkl")
        test=scalers.transform(unit)

        test = test.dropna()
        test = test.drop('Loan_ID', axis = 1)
        test['LoanAmount'] = (test['LoanAmount'] * 1000).astype('int')

        test_y = test['Loan_Status']
        test_y = test_y.map({'Y':1, 'N':0})

        test_x = test.drop('Loan_Status', axis = 1)
        test_x = pd.get_dummies(test_x)

        sc = MinMaxScaler()
        test_x = sc.fit_transform(test_x)

        y_pred = mdl.predict(test_x)
        y_pred=(y_pred>0.58)
        newdf=pd.DataFrame(y_pred, columns=['Status'])
        newdf=newdf.replace({True:'Approved', False:'Rejected'})
        return JsonResponse('Your Status is {}'.format(newdf), safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
'''