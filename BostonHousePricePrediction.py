import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from sklearn.ensemble import RandomForestRegressor
menu=st.sidebar.radio('menu',['Home','data','Visualization','Price Prediction'])
df=pd.read_csv('houseprice.csv')
if menu=='Home':
    st.title(" House Price Prediction App")
    st.write('---')
    st.image('house.jpg',width=550)

if menu=='data':
    df=pd.read_csv('houseprice.csv')
    st.header('data of housing prices')
    if st.checkbox('data'):
        st.write(df.sample(10))
    st.header('the shape of the data')
    if st.checkbox('shape'):
        st.write(df.shape)    
    st.header('statistical summary of the data')
    if st.checkbox('statistics'):
        st.write(df.describe())
if menu=='Visualization':
    st.header('Graphs') 
    graph=st.selectbox('Graphs Types',['scatter plot','bar plot','histogram','correlation'])
    if graph=='scatter plot':
        st.subheader('relation between crime rate and median value')
        plt.figure(figsize=(15,8)) 
        fig=px.scatter(data_frame=df, x='crim', y='medv',color='age')  
        st.plotly_chart(fig)

        st.subheader('relation between number of rooms and median value')
        plt.figure(figsize=(15,8)) 
        fig=px.scatter(data_frame=df, x='rm', y='medv',color='age')  
        st.plotly_chart(fig)

        st.subheader('relation between tax rate and median value')
        plt.figure(figsize=(15,8)) 
        fig=px.scatter(data_frame=df, x='tax', y='medv',color='age')  
        st.plotly_chart(fig)

        st.subheader('relation between lower status of the population and median value')
        plt.figure(figsize=(15,8)) 
        fig=px.scatter(data_frame=df, x='lstat', y='medv',color='age')  
        st.plotly_chart(fig)

    if graph=='bar plot':
        st.subheader('counts of accessibility to radial highways')
        fig=plt.figure(figsize=(15,8))
        sns.countplot(data=df,x='rad') 
        plt.xlabel('radial highway',fontsize=20)
        plt.ylabel('count',fontsize=20) 
        st.pyplot(fig)

        st.subheader('counts of tract river bounds')
        fig=plt.figure(figsize=(15,8))
        sns.countplot(data=df,x='chas') 
        plt.xlabel('river bounds',fontsize=20)
        plt.ylabel('count',fontsize=20) 
        st.pyplot(fig)

    if graph=='histogram':
        st.subheader('age distribution')
        fig=plt.figure(figsize=(15,8))
        sns.histplot(data=df,x='age',bins=12) 
        plt.xlabel('age',fontsize=20)
        plt.ylabel('count',fontsize=20) 
        st.pyplot(fig)

        st.subheader('median average distribution')
        fig=plt.figure(figsize=(15,8))
        sns.histplot(data=df,x='medv',bins=12) 
        plt.xlabel('median average',fontsize=20)
        plt.ylabel('count',fontsize=20) 
        st.pyplot(fig)

    if graph=='correlation':
       st.header('correlations')
       fig=plt.figure(figsize=(35,12))
       sns.heatmap(df.corr(),annot=True)
       st.pyplot(fig)

if menu=='Price Prediction':
    st.sidebar.header('input parameters')
    x=df.drop('medv',axis=1)
    y=df['medv']
    def input_features():
        crim=st.sidebar.slider('per capita crime rate by town',0.0,100.0)
        zn=st.sidebar.slider('proportion of residential land zoned for lots over 25,000 sq.ft',0.0,100.0)
        indus=st.sidebar.slider('proportion of non-retail business acres per town',0.0,50.0)
        chas=st.sidebar.slider('Charles River bounds',0.0,1.0)
        nox=st.sidebar.slider('nitrogen oxides concentration',0.0,1.0)
        rm=st.sidebar.slider('average number of rooms per dwelling',0.0,10.0)
        age=st.sidebar.slider('proportion of owner-occupied units built prior',0.0,100.0)
        dis=st.sidebar.slider('weighted mean of distances to five Boston employment centres.',0.0,20.0)
        rad=st.sidebar.slider('index of accessibility to radial highways.',0.0,50.0)
        tax=st.sidebar.slider('full-value property-tax rate per \$10,000.',0.0,1000.0)
        ptratio=st.sidebar.slider('pupil-teacher ratio by town.',0.0,50.0)
        black=st.sidebar.slider('proportion of blacks by town.',0.0,500.0)
        lstat=st.sidebar.slider('lower status of the population',0.0,50.0)
        data= {
        'crim': crim,'zn': zn,'indus': indus,'chas': chas,'nox': nox,
        'rm': rm,'age': age,'dis':dis,'rad':rad,'tax':tax,'ptratio': ptratio,'black': black,'lstat': lstat}
        features=pd.DataFrame(data,index=[0])
        return features
    df=input_features()
    st.header('input parametrs')
    st.write(df)
    st.write('---')
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor()
    model.fit(x,y)
    prediction=model.predict(x)
    st.header('median preiction')
    st.write(prediction[0])
    st.write('---')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    explainer=shap.TreeExplainer(model)
    shap_values=explainer.shap_values(x)
    st.header('features_importance')
    plt.title('features importance by shap')
    shap.summary_plot(shap_values,x)
    st.pyplot(bbox_inches='tight')
    st.write('---')
    
    shap.summary_plot(shap_values,x,plot_type='bar')
    st.pyplot(bbox_inches='tight')


        

