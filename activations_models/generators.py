from candidate_models.base_models.unsupervised_vvs import ModelBuilder


def unsup_vvs_generator(debug=False):
    vvs_models = ['resnet18-supervised', 'resnet18-la', 'resnet18-ir', 'resnet18-ae',
                  'resnet18-cpc', 'resnet18-color', 'resnet18-rp', 'resnet18-depth',
                  'resnet18-simclr', 'resnet18-deepcluster', 'resnet18-cmc']
    tf_res18_layers = ['encode_1.conv'] + ['encode_%i' % i for i in range(1, 10)]
    pt_resnet18_layers = ['relu', 'maxpool'] +\
                         ['layer1.0.relu', 'layer1.1.relu'] +\
                         ['layer2.0.relu', 'layer2.1.relu'] +\
                         ['layer3.0.relu', 'layer3.1.relu'] +\
                         ['layer4.0.relu', 'layer4.1.relu']
    prednet_layers = ['A_%i' % i for i in range(1, 4)] \
                     + ['Ahat_%i' % i for i in range(1, 4)] \
                     + ['E_%i' % i for i in range(1, 4)] \
                     + ['R_%i' % i for i in range(1, 4)]

    for model_identifier in vvs_models:
        activations_model = ModelBuilder()(model_identifier)

        if model_identifier in ModelBuilder.PT_MODELS:
            layers = pt_resnet18_layers
        elif model_identifier == 'prednet':
            layers = prednet_layers
        elif model_identifier == 'resnet18-simclr':
            layers = tf_res18_layers[1:]
        else:
            layers = tf_res18_layers

        yield activations_model, layers

        if debug:
            break
