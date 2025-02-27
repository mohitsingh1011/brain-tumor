# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import cv2
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tqdm import tqdm
# import os
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
# from sklearn.metrics import classification_report,confusion_matrix
# import ipywidgets as widgets
# import io
# from PIL import Image
# from IPython.display import display,clear_output
# from warnings import filterwarnings
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
# colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
# colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

# sns.palplot(colors_dark)
# sns.palplot(colors_green)
# sns.palplot(colors_red)

# labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

# X_train = []
# y_train = []
# image_size = 150
# for i in labels:
#     folderPath = os.path.join('../input/brain-tumor-classification-mri','Training',i)
#     for j in tqdm(os.listdir(folderPath)):
#         img = cv2.imread(os.path.join(folderPath,j))
#         img = cv2.resize(img,(image_size, image_size))
#         X_train.append(img)
#         y_train.append(i)
        
# for i in labels:
#     folderPath = os.path.join('../input/brain-tumor-classification-mri','Testing',i)
#     for j in tqdm(os.listdir(folderPath)):
#         img = cv2.imread(os.path.join(folderPath,j))
#         img = cv2.resize(img,(image_size,image_size))
#         X_train.append(img)
#         y_train.append(i)
        
# X_train = np.array(X_train)
# y_train = np.array(y_train)

# k=0
# fig, ax = plt.subplots(1,4,figsize=(20,20))
# fig.text(s='Sample Image From Each Label',size=18,fontweight='bold',
#              fontname='monospace',color=colors_dark[1],y=0.62,x=0.4,alpha=0.8)
# for i in labels:
#     j=0
#     while True :
#         if y_train[j]==i:
#             ax[k].imshow(X_train[j])
#             ax[k].set_title(y_train[j])
#             ax[k].axis('off')
#             k+=1
#             break
#         j+=1

# X_train, y_train = shuffle(X_train,y_train, random_state=101)

# X_train.shape
# X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)
# y_train_new = []
# for i in y_train:
#     y_train_new.append(labels.index(i))
# y_train = y_train_new
# y_train = tf.keras.utils.to_categorical(y_train)


# y_test_new = []
# for i in y_test:
#     y_test_new.append(labels.index(i))
# y_test = y_test_new
# y_test = tf.keras.utils.to_categorical(y_test)

# effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))

# model = effnet.output
# model = tf.keras.layers.GlobalAveragePooling2D()(model)
# model = tf.keras.layers.Dropout(rate=0.5)(model)
# model = tf.keras.layers.Dense(4,activation='softmax')(model)
# model = tf.keras.models.Model(inputs=effnet.input, outputs = model)

# model.summary()

# model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])

# tensorboard = TensorBoard(log_dir = 'logs')
# checkpoint = ModelCheckpoint("effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
#                               mode='auto',verbose=1)

# history = model.fit(X_train,y_train,validation_split=0.1, epochs =12, verbose=1, batch_size=32,
#                    callbacks=[tensorboard,checkpoint,reduce_lr])


# model.save('effnet.h5')

# filterwarnings('ignore')

# epochs = [i for i in range(12)]
# fig, ax = plt.subplots(1,2,figsize=(14,7))
# train_acc = history.history['accuracy']
# train_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']

# fig.text(s='Epochs vs. Training and Validation Accuracy/Loss',size=18,fontweight='bold',
#              fontname='monospace',color=colors_dark[1],y=1,x=0.28,alpha=0.8)

# sns.despine()
# ax[0].plot(epochs, train_acc, marker='o',markerfacecolor=colors_green[2],color=colors_green[3],
#            label = 'Training Accuracy')
# ax[0].plot(epochs, val_acc, marker='o',markerfacecolor=colors_red[2],color=colors_red[3],
#            label = 'Validation Accuracy')
# ax[0].legend(frameon=False)
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy')

# sns.despine()
# ax[1].plot(epochs, train_loss, marker='o',markerfacecolor=colors_green[2],color=colors_green[3],
#            label ='Training Loss')
# ax[1].plot(epochs, val_loss, marker='o',markerfacecolor=colors_red[2],color=colors_red[3],
#            label = 'Validation Loss')
# ax[1].legend(frameon=False)
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Training & Validation Loss')

# fig.show()

# pred = model.predict(X_test)
# pred = np.argmax(pred,axis=1)
# y_test_new = np.argmax(y_test,axis=1)

# print(classification_report(y_test_new,pred))

# fig,ax=plt.subplots(1,1,figsize=(14,7))
# sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True,
#            cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
# fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
#              fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

# plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Set the dataset path
# local_dataset_path = './dataset/Brain-Tumor-Classification-DataSet'
local_dataset_path = './dataset/Brain-Tumor-Classification-DataSet'



colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

sns.palplot(colors_dark)
sns.palplot(colors_green)
sns.palplot(colors_red)

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

X_train = []
y_train = []
image_size = 150


for i in labels:
    folderPath = os.path.join(local_dataset_path, 'Training', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)

for i in labels:
    folderPath = os.path.join(local_dataset_path, 'Testing', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)


# # Load training data
# for i in labels:
#     folderPath = os.path.join(local_dataset_path, 'Training', i)
#     for j in tqdm(os.listdir(folderPath)):
#         img = cv2.imread(os.path.join(folderPath, j))
#         img = cv2.resize(img, (image_size, image_size))
#         X_train.append(img)
#         y_train.append(i)

# # Load testing data
# for i in labels:
#     folderPath = os.path.join(local_dataset_path, 'Testing', i)
#     for j in tqdm(os.listdir(folderPath)):
#         img = cv2.imread(os.path.join(folderPath, j))
#         img = cv2.resize(img, (image_size, image_size))
#         X_train.append(img)
#         y_train.append(i)

X_train = np.array(X_train)
y_train = np.array(y_train)

k = 0
fig, ax = plt.subplots(1, 4, figsize=(20, 20))
fig.text(s='Sample Image From Each Label', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=0.62, x=0.4, alpha=0.8)
for i in labels:
    j = 0
    while True:
        if y_train[j] == i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k += 1
            break
        j += 1

X_train, y_train = shuffle(X_train, y_train, random_state=101)

X_train.shape
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=101)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4, activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs=model)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs')
# checkpoint = ModelCheckpoint("effnet.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
checkpoint = ModelCheckpoint("effnet.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001,
                              mode='auto', verbose=1)

history = model.fit(X_train, y_train, validation_split=0.1, epochs=12, verbose=1, batch_size=32,
                    callbacks=[tensorboard, checkpoint, reduce_lr])

model.save('effnet.h5')

epochs = [i for i in range(12)]
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

fig.text(s='Epochs vs. Training and Validation Accuracy/Loss', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=1, x=0.28, alpha=0.8)

sns.despine()
ax[0].plot(epochs, train_acc, marker='o', markerfacecolor=colors_green[2], color=colors_green[3],
           label='Training Accuracy')
ax[0].plot(epochs, val_acc, marker='o', markerfacecolor=colors_red[2], color=colors_red[3],
           label='Validation Accuracy')
ax[0].legend(frameon=False)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

sns.despine()
ax[1].plot(epochs, train_loss, marker='o', markerfacecolor=colors_green[2], color=colors_green[3],
           label='Training Loss')
ax[1].plot(epochs, val_loss, marker='o', markerfacecolor=colors_red[2], color=colors_red[3],
           label='Validation Loss')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Training & Validation Loss')

fig.show()

pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
y_test_new = np.argmax(y_test, axis=1)

print(classification_report(y_test_new, pred))

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(confusion_matrix(y_test_new, pred), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
           cmap=colors_green[::-1], alpha=0.7, linewidths=2, linecolor=colors_dark[3])
fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=0.92, x=0.28, alpha=0.8)

plt.show()
