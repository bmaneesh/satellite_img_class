from datetime import datetime
import numpy as np
from keras import metrics
import os
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras import losses
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint, Callback
from data_helper import AmazonPreprocessor, get_jpeg_data_files_paths
from metrics_udf import fmeasure, precision, recall
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_labels(y_pred, y_map):
    labels = []
    lab = []
    pred_mat = y_pred
    (x, y) = np.where(pred_mat>0.5)
    for idx in range(y.shape[0]):
        labels.append(y_map[y[idx]])
    labels = np.array(labels)
    for i in range(np.max(x)):
        lab.append([])
    for i in range(np.max(x)):
        lab[i] = labels[np.where(x==i)].tolist()
    return (x, y, lab)

BATCH_SIZE = 64
nb_epochs = 200
time  = datetime.now()
log_dir = '../logs/' + time.strftime('%Y%m%d-%H%M%S') + '/'

df = pd.read_csv('../train_v2.csv')
all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
data_dist = []
data_dist.append([x[1] for x in pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().values.tolist()])
data_weight = [1.0/x for x in data_dist[0]]
den = sum(data_weight)
data_weight = [x/den for x in data_weight]

[train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file] = get_jpeg_data_files_paths()
dataset = AmazonPreprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_jpeg_additional, img_resize=(224, 224))
dataset.init()

initial_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='max')
model_out = initial_model.output
model_out = Dense(256, activation='relu', input_shape=initial_model.output_shape[1:])(model_out)

out = []

interm_fc_layer = Dense(128, activation='relu')(model_out)
out.append(Dense(17, activation='sigmoid')(interm_fc_layer))

model = Model(initial_model.input, out)
for layer in model.layers:
    if 'conv' in layer.name:
        layer.trainable = False

# print model.summary()
optimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.1, nesterov=True)
model.compile(optimizer, loss=losses.binary_crossentropy, metrics=['binary_accuracy',precision, recall, fmeasure])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
tboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
stop_criteria = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='../checkpoints/weights.hdf5', verbose=1, save_best_only=True)

history = model.fit_generator(generator = dataset.get_train_generator(BATCH_SIZE),steps_per_epoch = int(len(dataset.X_train)/BATCH_SIZE), epochs=nb_epochs, verbose=0,
                              validation_data= dataset.get_val_generator(BATCH_SIZE), validation_steps=int(len(dataset.X_val)/BATCH_SIZE), callbacks=[lr_scheduler, tboard, stop_criteria, checkpointer])

test_preds = model.predict_generator(generator = dataset.get_prediction_generator(BATCH_SIZE), steps=int(len(dataset.X_test)/BATCH_SIZE), verbose=1)
(sample, _, labels) = test_labels(test_preds, dataset.y_map)
for i in range(10000):
    print labels[i]
np.savez('test_labels.npz', labels)

# {0: 'agriculture', 1: 'artisinal_mine', 2: 'bare_ground', 3: 'blooming', 4: 'blow_down', 5: 'clear', 6: 'cloudy', 7: 'conventional_mine', 8: 'cultivation',
#  9: 'habitation', 10: 'haze', 11: 'partly_cloudy', 12: 'primary', 13: 'road', 14: 'selective_logging', 15: 'slash_burn', 16: 'water'}
