import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)



def confirmed_cases(location_data, show=True, save_path='./graphs/world_map.jpg', save=True):
    """
    Creates a map of the world with radius of the circle representing the number
    of confirmed cases and the color representing the rate of fatality.
    """

    plt.figure(figsize=(20, 10))
    plt.title('Map of Confirmed Cases Across the World')
    sizes = location_data['Confirmed']**.5
    sc = plt.scatter(x=location_data['Long_'], y=location_data['Lat'],
                     s=sizes, c=location_data['Case-Fatality_Ratio'], alpha=0.4)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(*sc.legend_elements('sizes', 5),
               title='Number of Confirmed Cases$^{0.5}$')
    plt.colorbar(label='Fatility/Case Ratio')
    if not show:
        plt.close()
    if save:
        plt.savefig(save_path)

def most_confirmed_cases(location_data, show=True, save=True, save_path='./graphs/most_cases.jpg'):
    country_total = location_data.groupby(by='Country_Region').sum()
    confirmed_total = country_total.sort_values(by='Confirmed', ascending=True).tail(10).reset_index()
    plt.barh( y=confirmed_total['Country_Region'], width=confirmed_total['Confirmed'], color=CB91_Green)
    plt.xlabel('Number of confirmed cases')
    plt.ylabel('Country names')
    plt.title('Top countries with the most number of confirmed cases')
    if save:
        plt.savefig(save_path)
    if not show:
        plt.close()

def most_deaths(location_data, show=True, save=True, save_path='./graphs/most_deaths.jpg'):
    country_total = location_data.groupby(by='Country_Region').sum()
    confirmed_total = country_total.sort_values(by='Deaths', ascending=True).tail(10).reset_index()
    plt.barh( y=confirmed_total['Country_Region'], width=confirmed_total['Deaths'], color='#ECB296')
    plt.xlabel('Number of deaths')
    plt.ylabel('Country names')
    plt.title('Top countries with the most number of deaths')
    if save:
        plt.savefig(save_path)
    if not show:
        plt.close()

def most_incidence_rate(location_data, show=True, save=True, save_path='./graphs/most_incidence_rate.jpg'):
    country_total = location_data.groupby(by='Country_Region').mean()
    confirmed_total = country_total.sort_values(by='Incidence_Rate', ascending=True).tail(10).reset_index()
    plt.barh( y=confirmed_total['Country_Region'], width=confirmed_total['Incidence_Rate'], color='#127184')
    plt.xlabel('Incidence rate')
    plt.ylabel('Country names')
    plt.title('Top countries with the highest incidence rate')
    if save:
        plt.savefig(save_path)
    if not show:
        plt.close()

def most_fatality_rate(location_data, show=True, save=True, save_path='./graphs/most_fatality_ratio.jpg'):
    country_total = location_data.groupby(by='Country_Region').mean()
    confirmed_total = country_total.sort_values(by='Case-Fatality_Ratio', ascending=True).tail(10).reset_index()
    plt.barh( y=confirmed_total['Country_Region'], width=confirmed_total['Case-Fatality_Ratio'], color='#DB3823')
    plt.xlabel('Fatility ratio')
    plt.ylabel('Country names')
    plt.title('Top countries with the highest fatility ratio to confirmed cases')
    if save:
        plt.savefig(save_path)
    if not show:
        plt.close()



def male_female_cases(individual_cases, show=True, save=True, save_path='./graphs/gender_graph.jpg'):
    less_20 = individual_cases[individual_cases['age'] <= 20].groupby('sex').agg({'age':'count'}).to_numpy()
    twen_forty = individual_cases[(individual_cases['age']> 20) & (individual_cases['age']<= 40)].groupby('sex').agg({'age':'count'}).to_numpy()
    forty_sixty = individual_cases[(individual_cases['age']> 40) & (individual_cases['age']<= 60)].groupby('sex').agg({'age':'count'}).to_numpy()
    sixty_eighty = individual_cases[(individual_cases['age']> 60) & (individual_cases['age']<= 80)].groupby('sex').agg({'age':'count'}).to_numpy()
    eight_plus = individual_cases[individual_cases['age'] > 80].groupby('sex').agg({'age':'count'}).to_numpy()
    female = [less_20[0][0], twen_forty[0][0], forty_sixty[0][0], sixty_eighty[0][0], eight_plus[0][0]]
    male = [less_20[1][0], twen_forty[1][0], forty_sixty[1][0], sixty_eighty[1][0], eight_plus[1][0]]
    labels = ['','0-20', '21 - 40', '41-60', '61 - 80', '80+']  # the label locations
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(5)
    width = 0.2
    rects1 = ax.bar(x - width/2,male,width, label='Men')
    rects2 = ax.bar(x + width/2,female,width, label='Women')
    ax.set_ylabel('# of confirmed Cases')
    ax.set_xlabel('Age groups')
    ax.set_title('Number of confirmed cases by gender')
    ax.set_xticklabels(labels)
    ax.legend()
    if save:
        plt.savefig(save_path)
    if not show:
        plt.close()

