import json

import utils

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


def convert_examples_to_features(examples, vals_to_id, is_test=False):
    # Concatenate the one-hot repr. of each feature
    num_cols = 30 if is_test else 31
    features = []
    for eg in examples:
        e_id = int(eg[0])
        
        concat = []
        for i in range(num_cols-29, num_cols):
            if len(vals_to_id[i]) >= 100:
                continue
            feat = eg[i]
            val_id = vals_to_id[i][feat]
            onehot = get_onehot(val_id, len(vals_to_id[i]))
            concat += onehot
        feat = {
            'id': e_id,
            # 'label': label,
            'repr': concat,
        }
        if not is_test:
            feat['label'] = int(eg[1])
        features.append(feat)
    assert len(features[0]['repr']) == 157
    return features

examples = load_data('data/challenge1_train.csv')
cols = [get_col(examples, i) for i in range(0, 31)]
col_values = [set(c) for c in cols]
vals_to_id = []
for values in col_values:
    vals_to_id.append({v: i for i, v in enumerate(values)})
features = convert_examples_to_features(examples, vals_to_id, False)
utils.save_json_by_line(features, 'data/features.json')
examples = load_data('data/challenge1_test.csv')
vals_to_id = [vals_to_id[0]] + vals_to_id[2:]
features = convert_examples_to_features(examples, vals_to_id, True)
utils.save_json_by_line(features, 'data/test.json')

