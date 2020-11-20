from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import model_selection
from sklearn import preprocessing
import urllib.request
import pandas as pd
import numpy as np
import sklearn
import torch
import os

def download_files():
    if not os.path.exists('./data/joined.csv.gz'):
        data_url = 'https://github.com/alhparsa/CMPT459-Project/raw/main/data/joined.csv.gz'
        urllib.request.urlretrieve(data_url, './data/joined.csv.gz')
    if not os.path.exists('knn_model.pkl'):
        model_url = 'https://github.com/alhparsa/CMPT459-Project/raw/main/data/knn_model.pkl'
        urllib.request.urlretrieve(model_url, './data/knn_model.pkl')

class Data:
    def __init__(self):
        self.location = self.load_location_data()
        self.individual = self.load_individual_data()
        self.fix_countries()

    def fix_countries(self):
        self.individual.loc[self.individual.province ==
                            'Taiwan', 'country'] = "Taiwan"
        self.individual.loc[self.individual.country ==
                            'Puerto Rico', 'province'] = "Puerto Rico"
        self.individual.loc[self.individual.province ==
                            'Puerto Rico', 'country'] = "United States"
        self.location.loc[self.location['Country_Region']
                          == 'Taiwan*', 'Country_Region'] = 'Taiwan'
        self.location.loc[self.location['Country_Region'] ==
                          'Korea, South', 'Country_Region'] = 'South Korea'
        self.location.loc[self.location['Country_Region'] ==
                          'Czechia', 'Country_Region'] = 'Czech Republic'

    def load_location_data(self, path='./data/processed_location_Sep20th2020.csv', parse_date=True):
        return pd.read_csv(path)

    def load_individual_data(self, path='./data/processed_individual_cases_Sep20th2020.csv', parse_date=True):
        return pd.read_csv(path)

    @staticmethod
    def format_dates(x):
        """
        Returns a pandas date for our date values, if the date
        is a range, it will return the first valu.
        """
        try:
            if '-' in x:
                return pd.to_datetime(x.split('-')[0], dayfirst=True)
            else:
                return pd.to_datetime(x, dayfirst=True)
        except:
            return np.nan

    @staticmethod
    def format_age(x):
        """
        Converts the ages to float numbers, if the ages
        are in range format, it will return the mean of the
        two boundaries, and if it is months format it will
        convert it to years. 
        """
        try:
            return pd.to_numeric(x)
        except:
            if '-' in x:
                try:
                    return np.mean([int(i) for i in x.split('-')])
                except:
                    return int(x.split('-')[0])
            elif '+' in x:
                return int(x.split('+')[0])
            else:
                return int(x.split()[0])/12.

    @staticmethod
    def remove_outliers_loc(location_data):
        """
        Removing outliers that are coordinately impossible
        """
        loc = location_data.dropna(subset=['Lat', 'Long_'])
        loc = loc.drop(loc[(loc.Lat < -90) | (loc.Lat > 90)
                           | (loc.Long_ < -180) | (loc.Long_ > 180)].index)
        return loc

    @staticmethod
    def remove_outliers_ind(ind_data):
        """
        Removing outliers that are coordinately impossible
        """
        individual = ind_data.dropna(subset=['latitude', 'longitude'])

        # Remove values with out-of-bounds long/lat ranges
        individual = individual.drop(individual[(individual.latitude < -90) | (
            individual.latitude > 90) | (individual.longitude < -180) | (individual.longitude > 180)].index)

        return individual


class CleanedData:
    """
    Class used for milestone 2
    """

    def __init__(self, loc='./data/joined.csv.gz', test_ratio=0.2, impute_data=True, convert_non_numerical=False, normalize_data=False, **drop_columns):
        download_files()
        if 'gz' in loc:
            self.data = pd.read_csv(loc, compression='gzip', parse_dates=[6, 10],)
        else:
            self.data = pd.read_csv(loc, parse_dates=[6, 10],)
        self.data['confirmed_day'] = self.data['date_confirmation'].dt.dayofyear
        self.data.loc[self.data[self.data['additional_information'].str.contains(
            'contact', na=False)].index, 'in_contact'] = 1
        self.data.loc[self.data['in_contact'].isna(), 'in_contact'] = 0
        self.data = self.data.drop(columns=['additional_information'])
        self.encoder = sklearn.preprocessing.LabelEncoder()
        self.encoder.fit(self.data['outcome'])
        self.data['outcome'] = self.encoder.fit_transform(self.data['outcome'])
        self.data = self.data[self.data['Lat'].notna()]
        self.data.replace({np.nan: 'unknown'}, inplace=True)
        self.scaled_down = False
        if 'drop_columns' in drop_columns:
            if len(drop_columns['drop_columns']) > 0:
                self.data = self.data.drop(columns=list(drop_columns['drop_columns']))
        if convert_non_numerical:
            self.convert_data()
        if impute_data:
            self.impute_data()
        if normalize_data:
            self.scale_down_values()
        self.split_data(test_ratio)


    def encode_combined_key(self):
        self.combined_key_encoder = sklearn.preprocessing.LabelEncoder()
        self.combined_key_encoder.fit(
            np.array(self.data['Combined_Key']).reshape(-1, 1))
        self.data['transformed_Combined_Key'] = self.combined_key_encoder.fit_transform(
            self.data['Combined_Key'])

    def encode_sex(self):
        self.sex_encoder = sklearn.preprocessing.LabelEncoder()
        self.sex_encoder.fit(np.array(self.data['sex']).reshape(-1, 1))
        self.data['transformed_sex'] = self.sex_encoder.fit_transform(
            self.data['sex'])

    def scale_down_values(self):
        if self.scaled_down:
            return
        self.scaled_down = True
        self.data.loc[self.data[self.data['Incidence_Rate'] > 0].index, 'Incidence_Rate'] = np.log(
            self.data[self.data['Incidence_Rate'] > 0]['Incidence_Rate'].astype('float64'))
        self.data.loc[self.data[self.data['Confirmed'] > 0].index, 'Confirmed'] = np.log(
            self.data[self.data['Confirmed'] > 0]['Confirmed'].astype('float64'))
        self.data.loc[self.data[self.data['Deaths'] > 0].index, 'Deaths'] = np.log(
            self.data[self.data['Deaths'] > 0]['Deaths'].astype('float64'))
        self.data.loc[self.data[self.data['Recovered'] > 0].index, 'Recovered'] = np.log(
            self.data[self.data['Recovered'] > 0]['Recovered'].astype('float64'))
        self.data.loc[self.data[self.data['Active'] > 0].index, 'Active'] = np.log(
            self.data[self.data['Active'] > 0]['Active'].astype('float64'))

    def scale_up_values(self, data=None):
        if not self.scaled_down:
            return
        if data is None:
            self.scaled_down = False
            self.data.loc[self.data[self.data['Incidence_Rate'] > 0].index, 'Incidence_Rate'] = np.exp(
                self.data[self.data['Incidence_Rate'] > 0]['Incidence_Rate'])
            self.data.loc[self.data[self.data['Confirmed'] > 0].index, 'Confirmed'] = np.exp(
                self.data[self.data['Confirmed'] > 0]['Confirmed'])
            self.data.loc[self.data[self.data['Deaths'] > 0].index, 'Deaths'] = np.exp(
                self.data[self.data['Deaths'] > 0]['Deaths'])
            self.data.loc[self.data[self.data['Recovered'] > 0].index, 'Recovered'] = np.exp(
                self.data[self.data['Recovered'] > 0]['Recovered'])
            self.data.loc[self.data[self.data['Active'] > 0].index, 'Active'] = np.exp(
                self.data[self.data['Active'] > 0]['Active'])
        else:
            data.loc[data[data['Incidence_Rate'] > 0].index, 'Incidence_Rate'] = np.exp(
                data[data['Incidence_Rate'] > 0]['Incidence_Rate'])
            data.loc[data[data['Confirmed'] > 0].index, 'Confirmed'] = np.exp(
                data[data['Confirmed'] > 0]['Confirmed'])
            data.loc[data[data['Deaths'] > 0].index, 'Deaths'] = np.exp(
                data[data['Deaths'] > 0]['Deaths'])
            data.loc[data[data['Recovered'] > 0].index, 'Recovered'] = np.exp(
                data[data['Recovered'] > 0]['Recovered'])
            data.loc[data[data['Active'] > 0].index, 'Active'] = np.exp(
                data[data['Active'] > 0]['Active'])

    def impute_data(self):
        imputer = IterativeImputer(
            max_iter=100, random_state=50, missing_values=np.nan)
        data_ = self.data[['age', 'transformed_sex', 'transformed_Combined_Key', 'in_contact', 'confirmed_day', 'Case-Fatality_Ratio',
                           'Incidence_Rate', 'Active', 'Lat', 'Long_', 'Confirmed', 'Deaths', 'Recovered']].replace('unknown', np.nan)
        self.data[['age', 'transformed_sex', 'transformed_Combined_Key', 'in_contact', 'confirmed_day', 'Case-Fatality_Ratio',
                   'Incidence_Rate', 'Active', 'Lat', 'Long_', 'Confirmed', 'Deaths', 'Recovered']] = imputer.fit_transform(data_)

    def convert_data(self):
        self.encode_combined_key()
        self.encode_sex()

    def decode_combined_key(self, data=None):
        if data is None:
            self.data['Combined_Key'] = self.combined_key_encoder.inverse_transform(
                np.array(self.data['transformed_Combined_Key']).reshape(-1, 1))
        elif 'Combined_Key' in data.columns:
            data['Combined_Key'] = self.combined_key_encoder.inverse_transform(
                np.array(data['transformed_Combined_Key']).reshape(-1, 1))

    def decode_sex(self, data=None):
        if data is None:
            self.data['sex'] = self.sex_encoder.inverse_transform(
                np.array(self.data['transformed_sex']).reshape(-1, 1))
        elif 'transformed_sex' in data.columns:
            data['sex'] = self.sex_encoder.inverse_transform(
                np.array(data['transformed_sex']).reshape(-1, 1))

    def decode_outcome(self, data):
        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return self.encoder.inverse_transform(np.array(data, dtype='int8').reshape(-1, 1)).ravel()
        data_ = data.copy(deep=True)
        data_.loc[data.index] = self.encoder.inverse_transform(np.array(data.values, dtype='int8').reshape(-1, 1))
        return data_

    def convert_back(self, data=None):
        self.scale_up_values(data)
        self.decode_combined_key(data)
        self.decode_sex(data)

    def split_data(self, test_ratio):
        X = self.data.drop(
            columns=['outcome', 'sex', 'date_confirmation', 'Combined_Key'])
        y = self.data['outcome']
        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_ratio)

    @property
    def training_data(self):
        return self.X_train, self.y_train

    @property
    def test_data(self):
        return self.X_test, self.y_test
