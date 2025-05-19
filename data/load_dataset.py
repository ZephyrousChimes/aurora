import os
import pandas as pd
from datasets import Dataset, DatasetDict


def load_dataset():
    data_dir = f"{os.path.dirname(__file__)}/processed"

    train_data = []
    val_data = []
    test_data = []

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 1 - train_ratio - val_ratio

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath, parse_dates=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            item_id = os.path.splitext(filename)[0]
            start_date = df['Date'].iloc[0]
            total_len = len(df)

            # Indices to split the time series
            train_end = int(total_len * train_ratio)
            val_end = int(total_len * (train_ratio + val_ratio))

            # Dynamic features: [num_features x time_steps]
            dynamic_feats = df[['Open', 'High', 'Low', 'Volume']].T.values

            # Static feature
            feat_static_cat = None and [hash(item_id) % 1000]

            # Slicing helper
            def make_entry(start_idx, end_idx):
                return {
                    'start': str(df['Date'].iloc[start_idx]),
                    'target': df['Close'].iloc[start_idx:end_idx].tolist(),
                    'feat_dynamic_real': dynamic_feats[:, start_idx:end_idx].tolist(),
                    'feat_static_cat': feat_static_cat,
                    'item_id': item_id
                }

            train_data.append(make_entry(0, train_end))
            val_data.append(make_entry(train_end, val_end))
            test_data.append(make_entry(val_end, total_len))

    # Convert to Hugging Face DatasetDict
    ds_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

    print(ds_dict)

    return ds_dict
