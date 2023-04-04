import streamlit as st
import pandas as pd
import joblib
from dateutil.utils import today
from datetime import date


st.title("Used car's price prediction")
data_test=pd.read_csv('/Users/dmitrijmatienko/test.csv')
data_train=pd.read_csv('/Users/dmitrijmatienko/train.csv')

#Обработка загруженного файла
def preprocessing(data_train):
    data_train['saledate'] = data_train['saledate'].replace('PST', '', regex=True)
    data_train['saledate'] = data_train['saledate'].replace('PDT', '', regex=True)
    data_train['saledate'] = [date_str.replace('()', '') for date_str in data_train['saledate']]
    data_train['saledate'] = pd.to_datetime(data_train['saledate'], format='%a %b %d %Y %H:%M:%S GMT%z ', utc=True)
    data_train['saledate'] = pd.to_datetime(data_train['saledate'])
    data_train['saledate'] = data_train['saledate'].dt.year
    data_train['age_car'] = data_train['saledate'] - data_train['year']
    data_train = data_train.drop(['saledate', 'year'], axis=1)

    data_train['make'] = data_train['make'].str.lower()
    data_train['make'] = data_train['make'].replace('mercedes-benz', 'mercedes', regex=True)
    data_train['make'] = data_train['make'].replace('mercedes-b', 'mercedes', regex=True)
    data_train['make'] = data_train['make'].replace('mercedes-b', 'mercedes', regex=True)
    data_train['make'] = data_train['make'].replace('vw', 'volkswagen', regex=True)
    data_train['make'] = data_train['make'].replace('mazda tk', 'mazda', regex=True)
    data_train['make'] = data_train['make'].replace('dodge tk', 'dodge', regex=True)
    data_train['make'] = data_train['make'].replace('dot', 'dodge', regex=True)
    data_train['make'] = data_train['make'].replace('airstream', 'mercedes', regex=True)
    data_train['make'] = data_train['make'].replace('gmc truck', 'gmc', regex=True)
    data_train['make'] = data_train['make'].replace('landrover', 'land rover', regex=True)
    data_train['make'] = data_train['make'].fillna('other')

    data_train['model'] = data_train['model'].str.lower()
    data_train['model'] = data_train['model'].fillna('other')

    data_train['trim'] = data_train['trim'].str.lower()
    data_train['trim'] = data_train['trim'].fillna('other')

    data_train['body'] = data_train['body'].str.lower()
    data_train['body'] = data_train['body'].fillna('other')

    data_train['color'] = data_train['color'].replace('—', 'other', regex=True)
    data_train['color'] = data_train['color'].fillna('other')

    data_train['interior'] = data_train['interior'].replace('—', 'other', regex=True)
    data_train['interior'] = data_train['interior'].fillna('other')

    data_train['seller'] = data_train['seller'].str.lower()
    data_train['seller'] = data_train['seller'].fillna('other')

    data_train['transmission'] = data_train['transmission'].fillna('other')

    data_train['condition'] = data_train['condition'].fillna(data_train['condition'].mean())

    data_train['odometer'] = data_train['odometer'].fillna(data_train['odometer'].mean())

    return (data_train)#  #

#Обработка данных, введенных вручную
def preprocessing_one(data_train):
    data_train['saledate'] = pd.to_datetime(data_train['saledate'])
    data_train['saledate'] = data_train['saledate'].dt.year
    data_train['age_car'] = data_train['saledate'] - data_train['year']
    data_train = data_train.drop(['saledate', 'year'], axis=1)

    data_train['make'] = data_train['make'].str.lower()
    data_train['make'] = data_train['make'].replace('mercedes-benz', 'mercedes', regex=True)
    data_train['make'] = data_train['make'].replace('mercedes-b', 'mercedes', regex=True)
    data_train['make'] = data_train['make'].replace('mercedes-b', 'mercedes', regex=True)
    data_train['make'] = data_train['make'].replace('vw', 'volkswagen', regex=True)
    data_train['make'] = data_train['make'].replace('mazda tk', 'mazda', regex=True)
    data_train['make'] = data_train['make'].replace('dodge tk', 'dodge', regex=True)
    data_train['make'] = data_train['make'].replace('dot', 'dodge', regex=True)
    data_train['make'] = data_train['make'].replace('airstream', 'mercedes', regex=True)
    data_train['make'] = data_train['make'].replace('gmc truck', 'gmc', regex=True)
    data_train['make'] = data_train['make'].replace('landrover', 'land rover', regex=True)
    data_train['make'] = data_train['make'].fillna('other')

    data_train['model'] = data_train['model'].str.lower()
    data_train['model'] = data_train['model'].fillna('other')

    data_train['trim'] = data_train['trim'].str.lower()
    data_train['trim'] = data_train['trim'].fillna('other')

    data_train['body'] = data_train['body'].str.lower()
    data_train['body'] = data_train['body'].fillna('other')

    data_train['color'] = data_train['color'].replace('—', 'other', regex=True)
    data_train['color'] = data_train['color'].fillna('other')

    data_train['interior'] = data_train['interior'].replace('—', 'other', regex=True)
    data_train['interior'] = data_train['interior'].fillna('other')

    data_train['seller'] = data_train['seller'].str.lower()
    data_train['seller'] = data_train['seller'].fillna('other')

    data_train['transmission'] = data_train['transmission'].fillna('other')

    data_train['condition'] = data_train['condition'].fillna(data_train['condition'].mean())

    data_train['odometer'] = data_train['odometer'].fillna(data_train['odometer'].mean())

    return (data_train)

button = st.button('Show train data')

if button:
    st.write('Обзор обучающих данных до обработки')
    st.write(data_train.head(5))
    st.write('Количество пропущенных значений')
    st.write(data_train.isna().sum())
    st.write('Обзор численных признаков')
    st.write(data_train.describe())
    data_p = preprocessing(data_train)
    st.write('После обработки')
    st.write(data_p.head(5))
    st.write('Количество пропущенных значений')
    st.write(data_p.isna().sum())
    st.write('Обзор численных признаков')
    st.write(data_p.describe())

cat = joblib.load("/Users/dmitrijmatienko/model.pkl")


but=st.button('Обзор обученной модели')
if but:
    data_train = preprocessing(data_train)
    importance = pd.DataFrame({
        'Feature': data_train.drop(['sellingprice','vin'],axis=1).columns,
        'Catboost': cat.feature_importances_
    })
    importance = importance.sort_values(by="Catboost", ascending=False)

    st.write('Оценка важности признаков при обучении модели')
    st.write(importance)


st.write('Загрузите данные для предсказания')
upload_file=st.file_uploader('')
if upload_file:
    columns=['make','model','trim','body','transmission','state','color','interior','seller']
    file=pd.read_csv(upload_file)
    st.write(file.head(5))
    file=file.drop(['vin'],axis=1)
    predict=cat.predict(preprocessing(file))
    st.write(pd.Series(predict,name='sellingprice'))

st.write('Или введите данные вручную')
year = st.number_input('Year', min_value=1900, max_value=today().year, value=2014)
make = st.text_input('Make', value='Ford')
model = st.text_input('Model', value='Fusion')
trim = st.text_input('Trim', value='SE')
body = st.text_input('Body', value='Sedan')
transmission = st.radio('Transmission', ['automatic', 'manual'])
state = st.text_input('State', max_chars=2, value='mo')
condition = st.number_input('Condition', min_value=1.0, max_value=5.0, step=0.5, value=3.5)
odometer = st.number_input('Odometer', min_value=0, step=10000, value=31000)
color = st.text_input('Color', value='black')
interior = st.text_input('Interior', value='black')
seller = st.text_input('Seller', 'ars/avis budget group')
saledate = st.date_input('Sale date', value=date(2015, 2, 25))

btn_predict = st.button('Предсказать')

if btn_predict:
    df = pd.DataFrame({
        'year': [year],
        'make': [make],
        'model': [model],
        'trim': [trim],
        'body': [body],
        'transmission': [transmission],
        'state': [state],
        'condition': [condition],
        'odometer': [odometer],
        'color': [color],
        'interior': [interior],
        'seller': [seller],
        'saledate': [saledate],
    })
    predict = cat.predict(preprocessing_one(df)
    )
    st.write('Цена за автомобиль')
    st.write(float(predict))
