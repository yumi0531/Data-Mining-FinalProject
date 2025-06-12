import sys
sys.path.append('..')

import transtab

transtab.random_seed(42)

dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, binary_features = transtab.load_data('dataset')
print('########################################')

# warning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer was not TransformerEncoderLayer
model = transtab.build_classifier(categorical_features, numerical_features, binary_features, num_class=7)

print(model)
