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
from data_helper_all import AmazonPreprocessor, get_jpeg_data_files_paths
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

def f1measure(y_true, y_pred):
    # print K.int_shape(y_pred)
    y_pred = K.eval(y_pred)
    print 'hello1'
    y_true = K.eval(y_true)
    print 'hello2'
    f1 = f1_score(y_true, y_pred)
    print 'f1_score-{0}'.format(f1)
    return f1


# def recall(y_true, y_pred):
#     #Recall metric.
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),axis=0)
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)),axis=0)
#     print K.int_shape(possible_positives)
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def precision(y_true, y_pred):
#     #Precision metric.
#
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),axis=0)
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)),axis=1)
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
# #
# # def f1score(y_true, y_pred):
# #     print K.eval(y_pred)
# #     r = recall(y_true, y_pred)
# #     p = precision(y_true, y_pred)
# #     f1 = 2*(r*p)/(r+p+K.epsilon())
# #     return f1

# TODO CHANGE THE EPOCHS TO 100 epochs and run
# weights correspond to 100 epochs with sinteractive(latest in logs)
#weights_all correspond to 200 epochs with srun(latest-1 in logs)
BATCH_SIZE = 64
nb_epochs = 200
time  = datetime.now()
log_dir = '../logs/' + time.strftime('%Y%m%d-%H%M%S') + '/'

df = pd.read_csv('../train_v2.csv')
all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
# tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)
# print pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index()
data_dist = []
data_dist.append([x[1] for x in pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().values.tolist()])
# print 'data distribution-{0}'.format(data_dist[0])
data_weight = [1.0/x for x in data_dist[0]]
den = sum(data_weight)
data_weight = [x/den for x in data_weight]
# print 'data weight-{0}'.format(data_weight), len(data_weight)

[train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file] = get_jpeg_data_files_paths()
dataset = AmazonPreprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_jpeg_additional, img_resize=(224, 224))
dataset.init()

initial_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='max')
model_out = initial_model.output
model_out = Dense(256, activation='relu', input_shape=initial_model.output_shape[1:])(model_out)

out = []

interm_fc_layer = Dense(128, activation='sigmoid')(model_out)
out.append(Dense(17, activation='sigmoid')(interm_fc_layer))

model = Model(initial_model.input, out)
for layer in model.layers:
    if 'conv' in layer.name:
        layer.trainable = False

# print model.summary()
optimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.1, nesterov=True)
# optimizer = optimizers.Adam(lr=0.01)
model.compile(optimizer, loss=losses.binary_crossentropy, metrics=['binary_accuracy',precision, recall, fmeasure])#, loss_weights=data_weight), weighted_metrics = ['accuracy'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
tboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
stop_criteria = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='../checkpoints/weights_sbatch_temp.hdf5', verbose=1, save_best_only=True)

# auc = roc_auc_fun(dataset)

# TODO SAVE MODEL checkpoint
history = model.fit_generator(generator = dataset.get_train_generator(BATCH_SIZE),steps_per_epoch = int(len(dataset.X_train)/BATCH_SIZE), epochs=nb_epochs, verbose=0,
                              validation_data= dataset.get_val_generator(BATCH_SIZE), validation_steps=int(len(dataset.X_val)/BATCH_SIZE), callbacks=[lr_scheduler, tboard, stop_criteria, checkpointer])

# history = model.fit_generator(generator = dataset.get_train_generator(BATCH_SIZE),steps_per_epoch = int(len(dataset.X_train)/BATCH_SIZE), epochs=nb_epochs, verbose=0,
#                               validation_data= dataset.get_val_generator(int(len(dataset.X_train))), validation_steps=1, callbacks=[lr_scheduler, tboard, stop_criteria, checkpointer])
#
#
# # #
# # # history.history['loss'], history.history['val_loss']
test_preds = model.predict_generator(generator = dataset.get_prediction_generator(BATCH_SIZE), steps=int(len(dataset.X_test)/BATCH_SIZE), verbose=1)
# np.save('test_preds.npy', test_preds)
(sample, _, labels) = test_labels(test_preds, dataset.y_map)
for i in range(10000):
    print labels[i]
np.savez('test_labels.npz', labels)
# np.savez('history.npz', history)
# {0: 'agriculture', 1: 'artisinal_mine', 2: 'bare_ground', 3: 'blooming', 4: 'blow_down', 5: 'clear', 6: 'cloudy', 7: 'conventional_mine', 8: 'cultivation',
#  9: 'habitation', 10: 'haze', 11: 'partly_cloudy', 12: 'primary', 13: 'road', 14: 'selective_logging', 15: 'slash_burn', 16: 'water'}