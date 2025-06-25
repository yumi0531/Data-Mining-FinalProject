from torch import nn, optim

def get_parameter_names(model):
    return [
        *[
            f'{name}.{n}'
            for name, child in model.named_children()
            if not isinstance(child, nn.LayerNorm)
            for n in get_parameter_names(child)
        ],
        *[
            name
            for name in model._parameters.keys()
            if 'bias' not in name
        ]
    ]

def get_optimizer(model, learning_rate, weight_decay=0):
    decay_parameter_names = get_parameter_names(model)
    parameters = [
        {
            'params': [
                parameter
                for name, parameter in model.named_parameters()
                if name in decay_parameter_names
            ],
            'weight_decay': weight_decay
        },
        {
            'params': [
                parameter
                for name, parameter in model.named_parameters()
                if name not in decay_parameter_names
            ],
            'weight_decay': 0
        }
    ]
    optimizer = optim.Adam(parameters, learning_rate)
    return optimizer
