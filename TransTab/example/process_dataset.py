import os
import pandas

data = pandas.read_csv('../playground-series-s5e6/train.csv')

data = data.rename(columns={
    'Temparature': 'Temperature',
    'Fertilizer Name': 'target_label'
})
data['target_label'] = data['target_label'].map({
    '10-26-26': 0,
    '14-35-14': 1,
    '17-17-17': 2,
    '20-20': 3,
    '28-28': 4,
    'DAP': 5,
    'Urea': 6
})

numerical_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

os.makedirs('dataset', exist_ok=True)
with open('dataset/numerical_feature.txt', 'w') as file:
    file.write('\n'.join(numerical_features))
data.to_csv('dataset/data_processed.csv', index=False)

print('┌───────────────────────────────┐')
print('│ dataset/numerical_feature.txt │')
print('└───────────────────────────────┘')

with open('dataset/numerical_feature.txt') as file:
    print(file.read())

print('########################################')

print('┌────────────────────────────┐')
print('│ dataset/data_processed.csv │')
print('└────────────────────────────┘')

print(pandas.read_csv('dataset/data_processed.csv'))
