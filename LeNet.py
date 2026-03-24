from keras.models import Model
from keras.layers import *


def build_model(out_dim):
    input = Input(shape=(32, 32, 1), name='in')
    conv1_out = Conv2D(filters=6, kernel_size=(5, 5), activation='relu', name='c1')(input)
    maxp1_out = MaxPooling2D(pool_size=(2, 2), name='p1')(conv1_out)

    conv2_out = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', name='c2')(maxp1_out)
    maxp2_out = MaxPooling2D(pool_size=(2, 2), name='p2')(conv2_out)

    conv3_out = Conv2D(filters=120, kernel_size=(5, 5), activation='relu', name='c3')(maxp2_out)

    flat1_out = Flatten(name='Flat_1')(conv3_out)
    Fully1_out = Dense(units=84, name='f1')(flat1_out)
    output = Dense(units=out_dim, activation='softmax', name='out')(Fully1_out)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    return model
