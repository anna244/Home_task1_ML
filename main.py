# uvicorn main:app --reload

import pickle
import re
import typing

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fastapi import Body, FastAPI
from pydantic import BaseModel


with open('model.pickle', 'rb') as f: 
    data = pickle.load(f)

median_mileage = data['median_mileage']
median_engine = data['median_engine']
median_max_power = data['median_max_power']
median_seats = data['median_seats']
median_max_torque_rpm = data['median_max_torque_rpm']
median_max_torque = data['median_max_torque']

scaler: StandardScaler = data['scaler']
enc: OneHotEncoder = data['enc']
enc_feature_names: typing.List['str'] = data['enc_feature_names']
enc_name_model: OneHotEncoder = data['enc_name_model']
enc_name_model_feature_names: typing.List['str'] = data['enc_name_model_feature_names']
model: Ridge = data['model']
y_test_pred: typing.List[float] = data['y_test_pred']


item_example = {
    "name": "Mahindra Xylo E4 BS IV",
    "year": 2010,
    "km_driven": 168000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": "14.0 kmpl",
    "engine": "2498 CC",
    "max_power": "112 bhp",
    "torque": "260 Nm at 1800-2200 rpm",
    "seats": 7.0
}

item_example_list = [
    {
        "name": "Tata Nexon 1.5 Revotorq XE",
        "year": 2017,
        "km_driven": 25000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": "21.5 kmpl",
        "engine": "1497 CC",
        "max_power": "108.5 bhp",
        "torque": "260Nm@ 1500-2750rpm",
        "seats": 5.0
    }, {
        "name": "Honda Civic 1.8 S AT",
        "year": 2007,
        "km_driven": 218463,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner",
        "mileage": "12.9 kmpl",
        "engine": "1799 CC",
        "max_power": "130 bhp",
        "torque": "172Nm@ 4300rpm",
        "seats": 5.0
    }, {
        "name": "Honda City i DTEC VX",
        "year": 2015,
        "km_driven": 173000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": "25.1 kmpl",
        "engine": "1498 CC",
        "max_power": "98.6 bhp",
        "torque": "200Nm@ 1750rpm",
        "seats": 5.0 
    }
]


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

    # https://fastapi.tiangolo.com/tutorial/schema-extra-example/
    class Config:
        schema_extra = {
            "example": item_example
        }


def process_torque_column(df: pd.DataFrame):
    df['max_torque'] = df['torque'].str.extract('([\d.]+)Nm', flags=re.IGNORECASE)
    unprocessed_df = df[df['max_torque'].isna()][df['torque'].notna()]

    for index, item in unprocessed_df.iterrows():
        # обработаем строки вида 510@ 1600-2400
        r = re.findall('^[\d.]+', item['torque'], flags=re.IGNORECASE)
        if r:
            r = float(r[0])
            if 'kgm' in item['torque'].lower():
                # переводим из kgm в Nm
                r = r * 9.80665
            df.loc[index, 'max_torque'] = r
    df['max_torque'] = df['max_torque'].astype(float)

    # извлекаем rpm из строк вида "250Nm@ 1500-2500rpm" или "11.5@ 4,500(kgm@ rpm)""
    # ([\d,]+) - все числа и запятые
    # (?:\(kgm\@)?\s* - текст "(kgm@ ", вопрос после него ( \(kgm\@)? ) говорит о том, что его может и не быть
    # ?: - Группировка без обратной связи, см https://ru.wikipedia.org/wiki/Регулярные_выражения 
    df['max_torque_rpm'] = df['torque'].str.extract('([\d,]+)(?:(?:\(kgm\@)?\s*rpm)', flags=re.IGNORECASE)

    # строка 11.5@ 4,500(kgm@ rpm) превратится в 4,500, убираем запятые 
    df['max_torque_rpm'] = df['max_torque_rpm'].replace('[,]', '', regex = True)

    # проверим, из каких строк не смогли извлечь данные
    unprocessed_df = df[df['max_torque_rpm'].isna()][df['torque'].notna()]

    for index, item in unprocessed_df.iterrows():
        # обработаем строки вида 510@ 1600-2400
        r = re.findall('(?:@\s*)(?:\d+-)?(\d+)', item['torque'], flags=re.IGNORECASE)
        if not r:
            # 210 / 1900
            r = re.findall('(?:\d+\s*/\s*)(\d+)', item['torque'], flags=re.IGNORECASE)
        if r and len(r[0]) > 2:
            df.loc[index, 'max_torque_rpm'] = r[0]
    df['max_torque_rpm'] = df['max_torque_rpm'].astype(float)

    del df['torque']


def process_mileage_column(df: pd.DataFrame):
    for i in df['mileage']:
        if str(i).endswith('km/kg'):
           df['mileage'].replace(i, float(i[:-6])*1.40, inplace=True)

    df['mileage'] = df['mileage'].replace('[^\d\.]', '', regex = True)
    df['mileage'] = df['mileage'].astype(float)
    df.loc[ df['mileage'] < 0.1, 'mileage'] = np.nan


def process_engine_column(df: pd.DataFrame):
    df['engine'] = df['engine'].replace('[^\d\.]', '', regex = True)
    df['engine'] = df['engine'].astype(float)
    df.loc[df['engine'] < 0.1, 'engine'] = np.nan


def process_max_power_column(df: pd.DataFrame):
    df['max_power'] = df['max_power'].replace('[^\d\.]', '', regex = True) # убираем еденицы измерения
    df.loc[df['max_power'] == '', 'max_power'] = np.nan # в одной из строк была замечена пустая строка 
    df['max_power'] = df['max_power'].astype(float)
    df.loc[df['max_power'] < 0.1, 'max_power'] = np.nan # заменим все 0 на Nan


def apply_medians(df: pd.DataFrame):
    df['mileage'] = df['mileage'].fillna(median_mileage)
    df['engine'] = df['engine'].fillna(median_engine)
    df['max_power'] = df['max_power'].fillna(median_max_power)
    df['seats'] = df['seats'].fillna(median_seats)
    df['max_torque_rpm'] = df['max_torque_rpm'].fillna(median_max_torque_rpm)
    df['max_torque'] = df['max_torque'].fillna(median_max_torque)


def apply_feature_engineering (df: pd.DataFrame) -> pd.DataFrame:
    df['name_model'] = df['name'].str.extract('(^\S+)', flags=re.IGNORECASE)

    df["year^2_log"] = np.log(df["year"].pow(2)) # прологорифмируем те фичи что логнормально распределены
    df["km_driven^2_log"] = np.log(df["km_driven"].pow(2))
    df["mileage^2"] = df["mileage"].pow(2) #эту фичу не логарифмируем тк она имеет норм распредление

    df["pow_per_eng"] = df["max_power"]/df["engine"]
    df["engine_max_power"] = np.log(df["engine"]*df["max_power"])
    df["engine_max_torque"] = np.log(df["engine"]*df["max_torque"])

    df["year_log"] = np.log(df["year"])
    df["km_driven_log"] = np.log(df["km_driven"])
    df["max_power_log"] = np.log(df["max_power"])
    df["max_torque_log"] = np.log(df["max_torque"])

    df = df.drop(columns=['name', 'year','km_driven', 'max_power', 'max_torque'], axis = 1)

    columns_cat = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    column_name_model = ['name_model']
    columns_cat_all = columns_cat + column_name_model
    #columns_nums = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'max_torque','max_torque_rpm']
    columns_nums = [column for column in df.columns if column not in columns_cat_all]

    # нормировка данных
    scaled_features = scaler.transform(df[columns_nums])
    df = pd.concat(
        [
            df.loc[:, ~df.columns.isin(columns_nums)], 
            pd.DataFrame(data=scaled_features, columns=columns_nums)
        ], 
    axis=1)

    # OneHotEncoder для columns_cat
    codes = enc.transform(df[columns_cat]).toarray()
    df = pd.concat(
        [
            df.loc[:, ~df.columns.isin(columns_cat)], 
            pd.DataFrame(codes, columns=enc_feature_names).astype(int)
        ], 
    axis=1)

    # OneHotEncoder для name_model
    codes = enc_name_model.transform(df[column_name_model]).toarray()
    df = pd.concat(
        [
            df.loc[:, ~df.columns.isin(column_name_model)], 
            pd.DataFrame(codes, columns=enc_name_model_feature_names).astype(int)
        ], 
    axis=1)

    return df


def process_df(df:  pd.DataFrame) -> pd.DataFrame:
    process_torque_column(df)
    process_mileage_column(df)
    process_engine_column(df)
    process_max_power_column(df)
    apply_medians(df)

    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    
    return apply_feature_engineering(df)


def process_items(items: typing.List[Item]) -> pd.DataFrame:
    df = pd.DataFrame([model.dict() for model in items])
    return process_df(df)


def predict(df: pd.DataFrame):
    return np.exp(model.predict(df))


# проверим, что код выше работает
df_test = pd.read_csv('cars_test.csv')
del df_test['selling_price']
df_test = process_df(df_test)
y_pred = predict(df_test)
for index, item_saved in enumerate(y_test_pred):
    item_predict = y_pred[index]
    assert round(item_saved, 3) == round(item_predict, 3)


app = FastAPI()


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = process_items([item])
    y_pred = predict(df)
    return y_pred[0]


@app.post("/predict_items")
def predict_items(items: typing.List[Item] = Body(example=item_example_list)) -> typing.List[float]:
    df = process_items(items)
    return list(predict(df))
