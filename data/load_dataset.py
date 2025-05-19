import os
import pandas as pd
from datasets import Dataset, DatasetDict


def load_dataset():
    data_dir = f"{os.path.dirname(__file__)}/processed"

    prediction_length = 30  
    n_test_windows = 4      

    train_data = []
    val_data = []
    test_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath, parse_dates=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            item_id = os.path.splitext(filename)[0]
            feat_static_cat = None # [hash(item_id) % 1000]

            feat_dynamic_real = df[['Open', 'High', 'Low', 'Volume']].T.values
            full_target = df['Close'].values
            full_dates = df['Date'].values

            T = len(full_target)
            test_window = prediction_length * n_test_windows

            def make_entry(end_idx):
                return {
                    'start': str(full_dates[0]),
                    'target': full_target[:end_idx].tolist(),
                    'feat_dynamic_real': feat_dynamic_real[:, :end_idx].tolist(),
                    'feat_static_cat': feat_static_cat,
                    'item_id': item_id
                }

            train_data.append(make_entry(T - test_window))
            val_data.append(make_entry(T - prediction_length))
            test_data.append(make_entry(T))  

    # Create Hugging Face DatasetDict
    ds_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

    return ds_dict
