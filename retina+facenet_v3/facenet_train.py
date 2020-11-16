import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config.parameters import *
from facenet import InceptionResNetV2

def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

facenet_model = InceptionResNetV2()
input_shape = facenet_model.layers[0].input_shape[0][1:]

A = Input(shape=input_shape, name = 'anchor')
P = Input(shape=input_shape, name = 'anchorPositive')
N = Input(shape=input_shape, name = 'anchorNegative')

enc_A = facenet_model(A)
enc_P = facenet_model(P)
enc_N = facenet_model(N)


# Callbacks
early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00005)
checkpoint = ModelCheckpoint(filepath='./models/{epoch:03d}_{loss:.4f}_{val_loss:.4f}.h5',
                             monitor='val_loss',
                             verbose=1)

# Model
tripletModel = Model([A, P, N], [enc_A, enc_P, enc_N])
tripletModel.compile(optimizer = 'adam', loss = triplet_loss)

tripletModel.fit(gen, 
                 epochs=NUM_EPOCHS,
                 batch_size=32,
                 callbacks=[early_stopping, checkpoint])
