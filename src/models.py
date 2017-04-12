def nn(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    
    dnn = Flatten()(inputs)
    dnn1 = Dense(256)(dnn) 
    dnn2 = Dense(2)(dnn1)

    dnn2 = core.Activation('softmax')(dnn2)

    model = Model(input=inputs, output=dnn2)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def cnn(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    dnn = Flatten()(conv1) 
    conv2 = Dense(2)(dnn)

    conv2 = core.Activation('softmax')(conv2)

    model = Model(input=inputs, output=conv2)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model