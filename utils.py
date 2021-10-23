import json


def save_json_by_line(data, file):
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d))
            f.write('\n')


def load_json_by_line(file):
    data = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_acc(preds, labels):
    cnt = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            cnt += 1
    return cnt / len(preds)