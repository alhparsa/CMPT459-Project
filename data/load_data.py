import pandas as pd
import numpy as np


class Data:
    def __init__(self):
        self.location = self.load_location_data()
        self.individual = self.load_individual_data()
        self.fix_countries()
    
    def fix_countries(self):
        self.individual.loc[self.individual.province == 'Taiwan','country'] = "Taiwan"
        self.individual.loc[self.individual.country == 'Puerto Rico','province'] = "Puerto Rico"
        self.individual.loc[self.individual.province == 'Puerto Rico','country'] = "United States"
        self.location.loc[self.location['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
        self.location.loc[self.location['Country_Region'] == 'Korea, South', 'Country_Region'] = 'South Korea'
        self.location.loc[self.location['Country_Region'] == 'Czechia', 'Country_Region'] = 'Czech Republic'
    
    def load_location_data(self, path='./data/processed_location_Sep20th2020.csv',parse_date=True):
        return pd.read_csv(path)
    
    def load_individual_data(self, path='./data/processed_individual_cases_Sep20th2020.csv',parse_date=True):
        return pd.read_csv(path)
    
    @staticmethod
    def format_dates(x):
        """
        Returns a pandas date for our date values, if the date
        is a range, it will return the first valu.
        """
        try:
            if '-' in x:
                return pd.to_datetime(x.split('-')[0],dayfirst=True)
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
        loc = loc.drop(loc[(loc.Lat < -90) | (loc.Lat > 90) | (loc.Long_ < -180) | (loc.Long_ > 180)].index)
        return loc
    
    @staticmethod
    def remove_outliers_ind(ind_data):
        """
        Removing outliers that are coordinately impossible
        """
        individual = ind_data.dropna(subset=['latitude', 'longitude'])

        # Remove values with out-of-bounds long/lat ranges
        individual = individual.drop(individual[(individual.latitude < -90) | (individual.latitude > 90) | (individual.longitude < -180) | (individual.longitude > 180)].index)

        return individual