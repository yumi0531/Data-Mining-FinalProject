import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(filename, scaler=None, seed=42):
    data = pandas.read_csv(filename)
    data = data.rename(columns={'Temparature': 'Temperature'})
    data['Fertilizer Name'] = data['Fertilizer Name'].map({
        '10-26-26': 0,
        '14-35-14': 1,
        '17-17-17': 2,
        '20-20': 3,
        '28-28': 4,
        'DAP': 5,
        'Urea': 6
    })

    y = data['Fertilizer Name']
    x = data.drop(columns=['id', 'Fertilizer Name'])

    num_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    cat_features = ['Soil Type', 'Crop Type']

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y)
    valid_size = len(y) // 10
    valid_x = train_x.iloc[-valid_size:]
    valid_y = train_y[-valid_size:]
    train_x = train_x.iloc[:-valid_size]
    train_y = train_y[:-valid_size]

    if scaler is None:
        scaler = MinMaxScaler().fit(train_x[num_features])
    x[num_features] = scaler.transform(x[num_features])
    train_x[num_features] = scaler.transform(train_x[num_features])
    valid_x[num_features] = scaler.transform(valid_x[num_features])
    test_x[num_features] = scaler.transform(test_x[num_features])

    return (x, y), (train_x, train_y), (valid_x, valid_y), (test_x, test_y), cat_features, num_features, scaler

if __name__ == '__main__':
    dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, scaler = load_data('../playground-series-s5e6/train.csv')

    print('┌─────────┐')
    print('│ dataset │')
    print('└─────────┘')
    x, y = dataset
    print(x)
    print(y)

    print('════════════════════════════════════════')

    print('┌───────────────┐')
    print('│ train_dataset │')
    print('└───────────────┘')
    train_x, train_y = train_dataset
    print(train_x)
    print(train_y)

    print('════════════════════════════════════════')

    print('┌───────────────┐')
    print('│ valid_dataset │')
    print('└───────────────┘')
    valid_x, valid_y = valid_dataset
    print(valid_x)
    print(valid_y)

    print('════════════════════════════════════════')

    print('┌──────────────┐')
    print('│ test_dataset │')
    print('└──────────────┘')
    test_x, test_y = test_dataset
    print(test_x)
    print(test_y)

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
