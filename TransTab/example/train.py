import sys
sys.path.append('..')

import transtab

transtab.random_seed(42)

dataset, train_dataset, valid_dataset, test_dataset, categorical_features, numerical_features, binary_features = transtab.load_data('dataset', seed=42)
print('########################################')

model = transtab.build_classifier(categorical_features, numerical_features, binary_features, num_class=7)
print('########################################')

transtab.train(
    model,
    train_dataset,
    valid_dataset,
    num_epoch=5,
    batch_size=512,
    lr=1e-4,
    eval_metric='val_loss',
    output_dir='checkpoint',
    eval_less_is_better=True
)
