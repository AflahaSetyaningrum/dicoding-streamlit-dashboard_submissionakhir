import pickle

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)