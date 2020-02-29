import pandas as pd
import numpy as np
import re
import json
import pickle
from datetime import datetime as dt
from dateutil import parser
from treelib import Tree


def parse_raw_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    df = pd.DataFrame(columns=['name', 'birth_date', 'passport', 'occupation', 'chief', 'age', 'age_group',
                               'chief_level'])

    for line in lines:
        name = line[:find_nth_symbol(line, ' ', 3)]
        birth_date = re.search(r'\d{2}.\d{2}.\d{4}', line).group(0)
        passport = re.search(r'\d{10}', line).group(0)
        occupation = line[find_nth_symbol(line, ' ', 5) + 1: find_nth_symbol(line, ' ', 6)]
        if line[-1] == '\n':
            chief = line[find_nth_symbol(line, ' ', 6) + 1: -1]
        else:
            chief = line[find_nth_symbol(line, ' ', 6) + 1:]
        age = (dt.now().date() - parser.parse(birth_date).date()).days // 365

        df = df.append({'name': name,
                        'birth_date': birth_date,
                        'passport': passport,
                        'occupation': occupation,
                        'chief': chief,
                        'age': age}, ignore_index=True)

    df = df.drop_duplicates()
    df = set_age_group(df, lower_bound=AGE_LOWER_BOUND, upper_bound=AGE_UPPER_BOUND)

    return df


def find_nth_symbol(string, symbol, n):
    position = string.find(symbol)
    while n > 1:
        position = string.find(symbol, position + 1)
        n -= 1
    return position


def set_age_group(df, lower_bound, upper_bound):
    # Assuming people with the same age can't be in the different age groups
    for _, row in df.iterrows():
        if row['age'] <= np.percentile(df['age'], lower_bound):
            row['age_group'] = 'junior'
        elif row['age'] <= np.percentile(df['age'], upper_bound):
            row['age_group'] = 'middle'
        else:
            row['age_group'] = 'senior'
    return df


def set_chief_level(df):
    # Assuming chief level 0 is the highest
    df.set_index('name', inplace=True)
    while df['chief_level'].isnull().values.any():
        for _, row in df.iterrows():
            if row['chief'] == 'None':
                row['chief_level'] = 0
            elif df.loc[row['chief']]['chief_level'] is not pd.np.nan:
                row['chief_level'] = df.loc[row['chief']]['chief_level'] + 1
    return df


def make_tree(df):
    tree = Tree()
    df = set_chief_level(df)
    df = df.sort_values(by='chief_level')
    df.reset_index(inplace=True)

    for _, row in df.iterrows():
        if row['chief_level'] == 0:
            tree.create_node(tag=row['name'], identifier=row['name'])
        else:
            tree.create_node(tag=row['name'], identifier=row['name'], parent=row['chief'])
    return tree


def main(verbose=False):
    df = parse_raw_data(DATA_FILE)
    tree = make_tree(df)
    df = df.sort_values(by='name')
    df = df.reset_index(drop=False)
    jsn = df.to_json(orient='records', force_ascii=False)

    with open(OUTPUT_JSON_FILE, 'w+') as outfile:
        json.dump(jsn, outfile)
    with open(OUTPUT_PICKLE_FILE, 'wb+') as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        pd.set_option('display.max_columns', None)
        print(df, end='\n' * 2)
        tree.show()


if __name__ == '__main__':
    DATA_FILE = 'data.txt'
    OUTPUT_JSON_FILE = 'employees.json'
    OUTPUT_PICKLE_FILE = 'hierarchy.pkl'

    AGE_LOWER_BOUND = 33.33
    AGE_UPPER_BOUND = 66.66

    main()
