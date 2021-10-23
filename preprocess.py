import json

import matplotlib.pyplot as plt

import utils

file = 'data/challenge1_train.csv'

def load_data(file):
    with open(file, 'r') as f:
        next(f)
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(line.split(','))
    return data


def get_col(examples, num):
    col = []
    for eg in examples:
        col.append(eg[num])
    return col


def get_onehot(index, n):
    vec = [0] * n
    vec[index] = 1
    return vec


def convert_examples_to_features(examples):
    # Concatenate the one-hot repr. of each feature

    # Get one-hot mapping of each feature
    cols = [get_col(examples, i) for i in range(0, 31)]
    col_values = [set(c) for c in cols]
    vals_to_id = []
    for values in col_values:
        vals_to_id.append({v: i for i, v in enumerate(values)})
        # if len(vals_to_id[-1]) < 100:
        #     print(vals_to_id[-1])
    features = []
    for eg in examples:
        e_id = int(eg[0])
        label = int(eg[1])
        
        concat = []
        for i in range(2, 31):
            if i == 11:  # Skip f9 because it is equal to f11
                continue
            if len(vals_to_id[i]) >= 100:
                continue
            feat = eg[i]
            val_id = vals_to_id[i][feat]
            onehot = get_onehot(val_id, len(vals_to_id[i]))
            concat += onehot
        features.append({
            'id': e_id,
            'label': label,
            'repr': concat,
        })
    return features

examples = load_data(file)
features = convert_examples_to_features(examples)

file_features = 'data/features.json'
utils.save_json_by_line(features, file_features)
