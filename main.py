import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import minmax_scale,PolynomialFeatures
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
import plotly.graph_objs as go
import numpy as np
from scipy import stats
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime as dt

st.title("Stock Market Prediction by YATE ASSEKE RONALD")
symbols = tuple(pd.read_csv('Symbols.csv')['Symbols'].values)
stocks_name = st.sidebar.selectbox("Select your Stock",symbols)

start = '2010-01-01'
end = str(dt.now().date())

model_select = st.sidebar.selectbox('Select your Algorithm',('Linear Regression','Decision Tree'))
future = st.sidebar.slider('Select future period (days)',10,200,value=70)

def load_data(stock_name):
    data = yf.download(stock_name,start=start,end=end)
    data.dropna(axis=0,inplace=True)
    return data

data_load_message = st.text('Loading data .........')
data = load_data(stocks_name)
data_load_message.text("Loading data ......... Done")

st.subheader('Row Data')
st.write(data.tail())

def plotting(data):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index,y=data['Open'],name='Stock Open'))
    fig.add_trace(go.Scatter(x=data.index,y=data['Close'],name='Stock Close'))
    fig.layout.update(title_text=f'Open VS Close price of {stocks_name} Stock',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plotting(data)

def Daily_Return(data):

    daily_return =pd.DataFrame(data.Close.pct_change())
    daily_return.rename(columns={'Close':'Daily_return'},inplace=True)
    return daily_return

daily_return = Daily_Return(data)

st.subheader(f'Desciptive Statistics of Daily_Return of {stocks_name}')
daily_stats= daily_return['Daily_return'].describe().T
st.write(daily_stats)

def histplot(Daily_Return):
    fig = plt.figure()
    sns.distplot(Daily_Return['Daily_return'],color='darkorange')
    plt.xlabel('Daily return')
    plt.ylabel('Daily_return Frequency')
    plt.title("Distribution of Daily Return")
    st.pyplot(fig)

histplot(daily_return)

confidence_level = st.slider('Confidence level',min_value=0.0,max_value=1.0,step=0.01,value=0.8)
def confidence_mean_level(Daily_return,confidence_level):

    std = Daily_return['Daily_return'].std()
    mean = Daily_return['Daily_return'].mean()
    degree_of_freedom = len(Daily_return)-1
    num_observation = len(Daily_return)

    standard_error = std/np.sqrt(num_observation)
    t_student_score = stats.t.ppf(1-((1-confidence_level)/2),df=degree_of_freedom)

    Margin_error = standard_error * t_student_score
    skew = Daily_return['Daily_return'].skew()
    minimum = mean - Margin_error
    maximum = mean + Margin_error
    return minimum,maximum,skew

minimum,maximum,skewness = confidence_mean_level(daily_return,confidence_level)

st.write(f"We are {confidence_level*100}% confident that the mean interval of {stocks_name} Daily retain lies between \
                                {round(minimum,6)} and {round(maximum,6)} and the skewness is {skewness}")

def hyperparameters (model_select):

    paramateters = dict()
    if model_select == 'Linear Regression':
        pass

    elif model_select == 'Decision Tree':
        max_depth= st.sidebar.slider('Select max depth',1,30)
        min_samples_leaf = st.sidebar.slider('select minimum sample',5,60)
        paramateters['max_depth'] = max_depth
        paramateters['min_samples_leaf']= min_samples_leaf
    else:
        pass
    return paramateters

param = hyperparameters(model_select)


def models (model_select,param):

    if model_select == 'Linear Regression':
        models = LinearRegression()
    elif model_select == 'Decision Tree':
        models = DecisionTreeRegressor(max_depth=param['max_depth'],min_samples_leaf=param['min_samples_leaf'])
    else:
        pass
    return models


def get_train(datas,future):

    data = pd.DataFrame(datas['Close'])
    data['Prediction'] = data['Close'].shift(-future)

    x = data.drop('Prediction',1)[:-future]
    y = data.iloc[:-future,1]
    future = data.drop('Prediction',1).iloc[-future:]

    valid = pd.DataFrame(datas[x.shape[0]:]['Close'])

    return x,y,future,valid

my_model = models(model_select,param)

x,y,future,validation = get_train(data,future)

my_model.fit(x,y)

predict = my_model.predict(future)
validation['Prediction'] = predict

def plot_prediction (x,validation):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.index,y=x['Close'],name='Normal Data'))
    fig.add_trace(go.Scatter(x=validation.index,y=validation['Close'],name='Actual'))
    fig.add_trace(go.Scatter(x=validation.index,y=validation['Prediction'],name='Prediction'))
    fig.layout.update(title_text=f'Actual and predict value of  {stocks_name} Stock',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
st.write("""
          ### Prediction Part""")
plot_prediction(x,validation)

st.write(""" ### Performance """ )

mae = mean_absolute_error(validation['Close'].values,validation['Prediction'])
r2 = r2_score(validation['Close'].values,validation['Prediction'])

st.write(f'Mean Absolute Error : {mae}')
st.write(f'R Square : {r2}')




