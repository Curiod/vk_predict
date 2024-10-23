import pandas as pd
import numpy as np

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()
    features['length'] = df['values'].apply(lambda x: len(x)) ###
    features['mean'] = df['values'].apply(lambda x: x.mean())
    features['median'] = df['values'].apply(lambda x: np.median(x))
    features['zeros'] = df['values'].apply(lambda x:(x[:-1] * x[1:] < 0).sum()) / features['length']

    features['quantile_25'] = df['values'].apply(lambda x: np.percentile(x, 25)) 
    features['quantile_75'] = df['values'].apply(lambda x: np.percentile(x, 75))
    features['quantile_05'] = df['values'].apply(lambda x: np.percentile(x, 5)) 
    features['quantile_95'] = df['values'].apply(lambda x: np.percentile(x, 95))
    features['most_ac_lag'] = df['values'].apply(lambda x: find_best_lag(x, 12))
    features['ewm_12'] = df['values'].apply(lambda x: pd.Series(x[-36:]).ewm(span=12).mean().values[-1])
    features['ewmstd_12'] = df['values'].apply(lambda x: pd.Series(x[-36:]).ewm(span=12).std().values[-1])
    features['dom_freq'] = df.apply(lambda x: find_dominating_frequencies(x)[0], axis=1)
    features['dv_dt'] = df.apply(calculate_derivative, axis=1)
    features['mean_d'] = features['dv_dt'].apply(lambda x: x.mean())
    features['mean_1y'] = df['values'].apply(lambda x: x[:12].mean())

    return features.drop(['dv_dt', 'length'], axis=1)


def find_best_lag(data, max_lag):
    best_lag = 0
    best_correlation = 0

    for lag in range(1, max_lag + 1):
        shifted_data = np.roll(data, lag)
        
        correlation = np.corrcoef(data[lag:], shifted_data[lag:])[0, 1]

        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag

    return best_lag


def find_dominating_frequencies(row: pd.DataFrame, top_n=1):
    '''
    Returns top-n dominating frequencies in fft
    '''
    signal = row['values']
    fft = np.fft.fft(signal)
    time = row['dates']

    frequencies = np.fft.fftfreq(len(time), d=1)
    magnitude = np.abs(fft[frequencies>0]) 
    frequencies = frequencies[frequencies>0]
    

    ind = np.argpartition(magnitude, -top_n)[-top_n:]

    return frequencies[ind]


def calculate_derivative(df: pd.DataFrame, frame=0):
    values = df['values'][-frame:]
    dates = df['dates'][-frame:]
    dv_dt = np.array([(values[i+1]-values[i])/(dates[i+1].month - dates[i].month + 12 * (dates[i+1].year - dates[i].year)) 
            for i in range(len(values) - 1)])

    return dv_dt