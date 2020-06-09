import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import sklearn.mixture as sk

def load_csv_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    return df

def filter_by_attribute(df, attribute, value):
    return df[df[attribute] == value]

def get_cases_chronologically(df):
    cases = []
    labels = []
    for i in range(df.shape[0]):
        _cases = df.iloc[i, 12:]
        _labels = df.iloc[i, :12]
        cases.append(_cases)
        labels.append(_labels)
    
    cases = np.array(cases)
    labels = np.array(labels)
    return cases, labels

def TimeSinceHalfGenerator(caseseries):
    dubtime = []
    
    for idx, case in enumerate(caseseries):
        if case == 1 or case == 0 or (idx > 0 and caseseries[idx - 1] > caseseries[idx]):
            dubtime.append(0)
        else:
            appended = False
            for indx,x in enumerate(caseseries[0:idx]):
                if x == case / 2:
                    dubtime.append(idx-indx)
                    appended = True
                    break
                elif x > case / 2 and indx > 0 and caseseries[indx - 1] < case / 2:
                    arx = 1 - (((case/2) - caseseries[indx - 1])/(x - caseseries[indx - 1]))
                    dubtime.append(idx - indx + arx)
                    appended = True
                    break
                elif x > case / 2:
                    dubtime.append(idx-indx)
                    appended = True
                    break
            if not appended:
                dubtime.append(0)
    
    return np.array(dubtime)

def SimpleMovingAverage(npa, n):
    ret = np.cumsum(npa,dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
        



plt.style.use('fivethirtyeight')
BASE_PATH = '../COVID-19/csse_covid_19_data/'

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19-covid-confirmed.csv')
    

confirmed = load_csv_data("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
confirmedb = load_csv_data("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv")

color = ["red","orange","yellow","green","blue","purple","maroon","black","lime","brown","grey","khaki","lawngreen","peru","crimson","pink","cyan","lavender","darkslateblue","olive"]

"""df = filter_by_attribute(
        confirmedb, "Country_Region", "US")
cases, labels = get_cases_chronologically(df)
cases = cases.sum(axis=0)

bas50 = np.min(np.where(cases >= 50))

doubletime = TimeSinceHalfGenerator(cases)
smadt = SimpleMovingAverage(doubletime, 5)
diffdt = np.diff(doubletime)
smadiff = SimpleMovingAverage(diffdt,7)
plt.plot(range(len(doubletime)) - bas50,doubletime,label="US")"""
#plt.plot(np.array(range(len(smadiff)))+3.5,smadiff,label= "US")

i = 0
j=0
results = []
country = []
for val in np.unique(confirmed["Country/Region"]):
    df = filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = get_cases_chronologically(df)
    cases = cases.sum(axis=0)

    if val in ["Ecuador"]:
        bas50 = np.min(np.where(cases >= 50))
        doubletime = TimeSinceHalfGenerator(cases)#[bas50-1:]
        """if len(doubletime) < 60:
            alp = np.zeros((60-len(doubletime)))
            alp.fill(doubletime[-1])
            doubletime = np.concatenate((doubletime,alp))
        if len(doubletime) > 60:
            doubletime = doubletime[0:60]"""
            
        #plt.plot(range(len(doubletime)) - bas50,doubletime,label = val,color=color[i],linestyle="solid")
        diffdt = np.diff(doubletime)
        smadiff = SimpleMovingAverage(diffdt,7)
        plt.plot(np.array(range(len(smadiff)))+3.5-bas50,smadiff,label = val,color=color[i%20],linestyle="solid")

        #results.append(smadiff)
        #country.append(val)
        i+=1

    """if cases[-1] <= 5000 and cases[-1] > 1000:
        bas50 = np.min(np.where(cases >= 50))
        doubletime = TimeSinceHalfGenerator(cases)
        #plt.plot(range(len(doubletime)) - bas50,doubletime,label = val,color=color[j],linestyle="dashed")
        j+=1
        #smadt = SimpleMovingAverage(doubletime, 5)
        diffdt = np.diff(doubletime)
        smadiff = SimpleMovingAverage(diffdt,7)
        plt.plot(np.array(range(len(smadiff)))+3.5-bas50,smadiff,label = val,color=color[j],linestyle="dashed")"""

"""results = np.array(results)
country = np.array(country)
model = sk.GaussianMixture(n_components=5,covariance_type="spherical",max_iter = 1000)
trement = model.fit_predict(results)


for item in np.unique(trement):
    print(item)
    print(country[np.where(trement == item)])"""

#plt.plot(range(len(doubletime)),doubletime,color="blue")
#plt.plot(np.array(range(len(smadt)))+1.5,smadt,color="green")
#plt.plot(range(len(diffdt)),np.diff(doubletime),color="red")

plt.title("Rate of Death Doubling Time")
plt.xlabel("Days since 50 Deaths")
plt.ylabel("Doubling Time (Days)")
plt.legend()
plt.show()




