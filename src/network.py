from keras import Input, Model
from keras.applications import MobileNetV2
from keras.layers import Conv2D, Dropout, pooling, Dense

from gradient_flip import GradientReversal


def basenet(input_shape=(None, None, 3), n_conv=3,
            init_filter_size=10, dropout_rate=0.10,
            conv1x1_filters=None, include_top=False,
            hidden_units=None, n_classes=None):

    x = inp = Input(shape=input_shape)

    for i in range(0, n_conv, 1):
        x = Conv2D(filters=init_filter_size * (2**i),
                   kernel_size=3, activation='relu')(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        if i < n_conv - 1:
            x = pooling.MaxPool2D()(x)
    if conv1x1_filters:
        x = Conv2D(filters=conv1x1_filters,
                   kernel_size=1, activation='relu')(x)

    x = pooling.GlobalAvgPool2D()(x)

    if include_top:
        if isinstance(hidden_units, (list, tuple)):
            for units in hidden_units:
                x = Dense(units, activation='relu')(x)
        else:
            x = Dense(hidden_units, activation='relu')(x)
        x = Dense(n_classes, activation="sigmoid")(x)
    model = Model(inp, x)
    return model


def get_multihead_branch(inputs, num_classes, final_act, branch_name=None, dense=None,  gradflip=None):
    assert isinstance(dense, list) or dense is None

    x = inputs
    if gradflip:
        flip_layer = GradientReversal(gradflip)
        x = flip_layer(x)

    for d in dense:
        x = Dense(d, activation='relu')(x)
    x = Dense(num_classes, activation=final_act,
              name=branch_name or final_act)(x)
    return x


def get_numberonly_network(backbone, num_classes=10):
    number = get_multihead_branch(
        backbone.output, num_classes, final_act='softmax', branch_name='number', dense=[20])
    model = Model(backbone.input, [number], name='number_model')
    return model


def get_multitask_network(backbone, num_classes=10):
    number = get_multihead_branch(
        backbone.output, num_classes, final_act='softmax', branch_name='number', dense=[20])
    color = get_multihead_branch(
        backbone.output, num_classes, final_act='softmax', branch_name='color', dense=[20])
    model = Model(backbone.input, [number, color], name='number_color_model')
    return model


def get_multitask_network_gradflip(backbone, num_classes=10, gradflip_lambda=.1):
    outputs = backbone.output
    number = get_multihead_branch(
        outputs, num_classes, 'softmax', 'number', dense=[20])
    color = get_multihead_branch(
        outputs, num_classes, 'softmax', 'color', dense=[40, 20], gradflip=gradflip_lambda)

    model = Model(backbone.input, [number, color],
                  name='number_color_gradflip')

    return model
