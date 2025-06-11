import sys
sys.path.append('..')

import transtab

transtab.random_seed(42)

dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, binary_features = transtab.load_data('../dataset')
print('########################################')

print('┌─────────┐')
print('│ dataset │')
print('└─────────┘')
x, y = dataset
print(x)
print(y)

print('########################################')

print('┌───────────────┐')
print('│ train_dataset │')
print('└───────────────┘')
train_x, train_y = train_dataset
print(train_x)
print(train_y)

print('########################################')

print('┌───────────────┐')
print('│ valid_dataset │')
print('└───────────────┘')
valid_x, valid_y = valid_dataset
print(valid_x)
print(valid_y)

print('########################################')

print('┌──────────────┐')
print('│ test_dataset │')
print('└──────────────┘')
test_x, test_y = test_dataset
print(test_x)
print(test_y)

print('########################################')

print('┌──────────────────────┐')
print('│ categorical_features │')
print('└──────────────────────┘')
print(categorical_features)

print('########################################')

print('┌────────────────────┐')
print('│ numerical_features │')
print('└────────────────────┘')
print(numerical_features)

print('########################################')

print('┌─────────────────┐')
print('│ binary_features │')
print('└─────────────────┘')
print(binary_features)
