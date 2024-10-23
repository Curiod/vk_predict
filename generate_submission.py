import pickle
import pandas as pd

import os 
import warnings
warnings.filterwarnings("ignore")
from preprocess import generate_features


if __name__ == '__main__':
    test = pd.read_parquet(os.path.join('data', 'test.parquet'))
    features = generate_features(test)

    xgb_model = pickle.load(open(os.path.join('models', 'best_xgb.pkl'), 'rb'))
    probs = xgb_model.predict_proba(features)[:, 1]

    results = pd.DataFrame(columns=['id', 'score'])
    results['id'] = test['id']
    results['score'] = probs

    results.to_csv('submission.csv', index=False)