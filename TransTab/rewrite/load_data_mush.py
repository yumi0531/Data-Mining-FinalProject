import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
# Define mappings for categorical values
CATEGORY_MAPPINGS = {
    'cap-shape': {
        'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 
        's': 'sunken', 'p': 'spherical', 'o': 'others'
    },
    'cap-surface': {
        'i': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth',
        'h': 'shiny', 'l': 'leathery', 'k': 'silky', 't': 'sticky',
        'w': 'wrinkled', 'e': 'fleshy'
    },
    'cap-color': {
        'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
        'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow', 'l': 'blue',
        'o': 'orange', 'k': 'black'
    },
    'does-bruise-or-bleed': {
        't': 'bruises-or-bleeding', 'f': 'no'
    },
    'gill-attachment': {
        'a': 'adnate', 'x': 'adnexed', 'd': 'decurrent', 'e': 'free',
        's': 'sinuate', 'p': 'pores', 'f': 'none', '?': 'unknown'
    },
    'gill-spacing': {
        'c': 'close', 'd': 'distant', 'f': 'none'
    },
    'gill-color': {
        'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
        'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow', 'l': 'blue',
        'o': 'orange', 'k': 'black', 'f': 'none'
    },
    'stem-root': {
        'b': 'bulbous', 's': 'swollen', 'c': 'club', 'u': 'cup', 'e': 'equal',
        'z': 'rhizomorphs', 'r': 'rooted'
    },
    'stem-surface': {
        'i': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth',
        'h': 'shiny', 'l': 'leathery', 'k': 'silky', 't': 'sticky',
        'w': 'wrinkled', 'e': 'fleshy', 'f': 'none'
    },
    'stem-color': {
        'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
        'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow', 'l': 'blue',
        'o': 'orange', 'k': 'black', 'f': 'none'
    },
    'veil-type': {
        'p': 'partial', 'u': 'universal'
    },
    'veil-color': {
        'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
        'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow', 'l': 'blue',
        'o': 'orange', 'k': 'black', 'f': 'none'
    },
    'has-ring': {
        't': 'ring', 'f': 'none'
    },
    'ring-type': {
        'c': 'cobwebby', 'e': 'evanescent', 'r': 'flaring', 'g': 'grooved',
        'l': 'large', 'p': 'pendant', 's': 'sheathing', 'z': 'zone',
        'y': 'scaly', 'm': 'movable', 'f': 'none', '?': 'unknown'
    },
    'spore-print-color': {
        'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
        'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow', 'l': 'blue',
        'o': 'orange', 'k': 'black', 'f': 'none'
    },
    'habitat': {
        'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths',
        'h': 'heaths', 'u': 'urban', 'w': 'waste', 'd': 'woods'
    },
    'season': {
        's': 'spring', 'u': 'summer', 'a': 'autumn', 'w': 'winter'
    }
}


def load_data(filename, scaler=None, seed=42, missing_rate=0.5):
    data = pd.read_csv(filename, sep=';')
    
    # Map class label: 'e' → 1, 'p' → 0
    data['class'] = data['class'].map({'e': 1, 'p': 0})
    y = data['class']
    x = data.drop(columns=['class'])

    # random missing value
    np.random.seed(seed)
    mask = np.random.uniform(0, 1, (len(x), len(x.columns))) < missing_rate
    x[mask] = np.nan

    # Identify numerical and categorical features
    num_features = ['cap-diameter', 'stem-height', 'stem-width']
    cat_features = [col for col in x.columns if col not in num_features]

    # Replace categorical codes with CLIP-friendly text
    for col in cat_features:
        mapping = CATEGORY_MAPPINGS.get(col, {})
        x[col] = x[col].map(lambda v: mapping.get(v, v))

    # Split dataset
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    valid_size = len(train_y) // 10
    valid_x = train_x.iloc[-valid_size:]
    valid_y = train_y.iloc[-valid_size:]
    train_x = train_x.iloc[:-valid_size]
    train_y = train_y[:-valid_size]

    # Normalize numerical features
    if scaler is None:
        scaler = MinMaxScaler().fit(train_x[num_features])
    for df in [x, train_x, valid_x, test_x]:
        df[num_features] = scaler.transform(df[num_features])

    return (x, y), (train_x, train_y), (valid_x, valid_y), (test_x, test_y), cat_features, num_features, scaler


if __name__ == '__main__':
    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../MushroomDataset/secondary_data.csv')

    print('┌─────────┐')
    print('│ dataset │')
    print('└─────────┘')
    x, y = dataset
    print(x.head())
    print(y.head())

    print('════════════════════════════════════════')

    print('┌───────────────┐')
    print('│ train_dataset │')
    print('└───────────────┘')
    train_x, train_y = train_dataset
    print(train_x.head())
    print(train_y.head())

    print('════════════════════════════════════════')

    print('┌───────────────┐')
    print('│ valid_dataset │')
    print('└───────────────┘')
    valid_x, valid_y = valid_dataset
    print(valid_x.head())
    print(valid_y.head())

    print('════════════════════════════════════════')

    print('┌──────────────┐')
    print('│ test_dataset │')
    print('└──────────────┘')
    test_x, test_y = test_dataset
    print(test_x.head())
    print(test_y.head())

    print('════════════════════════════════════════')

    print('┌──────────────────────┐')
    print('│ categorical_features │')
    print('└──────────────────────┘')
    print(categorical_features)

    print('════════════════════════════════════════')

    print('┌────────────────────┐')
    print('│ numerical_features │')
    print('└────────────────────┘')
    print(numerical_features)