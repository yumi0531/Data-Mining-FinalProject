import sys
sys.path.append('..')

import transtab

transtab.random_seed(42)

def map_k(ys, predicts, k=3):
    labels = (-predicts).argsort(-1)[:,:k]
    map_k = 0
    for y, labels in zip(ys, labels):
        for index, label in enumerate(labels):
            if y == label:
                map_k += 1 / (index + 1)
                break
    map_k /= len(ys)
    return map_k

dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, binary_features = transtab.load_data('dataset', seed=42)
print('########################################')

model = transtab.build_classifier(categorical_features, numerical_features, binary_features, num_class=7, checkpoint='checkpoint')
print('########################################')

test_x, test_y = test_dataset
predicts = transtab.predict(model, test_x)

predict_labels = predicts.argmax(1)
map_3 = map_k(test_y, predicts)
accuracy = transtab.evaluate(predicts, test_y, metric='acc')

print('┌──────────┐')
print('│ predicts │')
print('└──────────┘')
print(predicts)
print(f'shape: {predicts.shape}')

print('########################################')

print('┌────────────────┐')
print('│ predict_labels │')
print('└────────────────┘')
print(predict_labels)
print(f'shape: {predict_labels.shape}')

print('########################################')

print('┌───────┐')
print('│ MAP@3 │')
print('└───────┘')
print(map_3)

print('########################################')

print('┌──────────┐')
print('│ accuracy │')
print('└──────────┘')
print(accuracy)
