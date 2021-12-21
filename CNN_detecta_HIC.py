# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:40:30 2021

@author: Larissa Lima
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling3D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_layer_normalization import LayerNormalization
from sklearn.model_selection import StratifiedKFold, train_test_split
from time import time
from sklearn.utils import class_weight
from tensorflow.python.keras import backend, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import datetime
import tweaked_ImageGenerator_v2 as imggen


def data_aug(entrada_x, entrada_y, n_dados):
  new_imgs = []
  new_label = []

  for paciente in range(len(entrada_x)):
    img = entrada_x[paciente].reshape(20,img_size,img_size)
    label = entrada_y[paciente]
    for rotacoes in range(n_dados):
      image = imggen.random_rotation(img, 360, fill_mode='nearest')
      new_imgs.append(np.array(image))
      new_label.append(label)
  new_imgs = np.array(new_imgs)
  new_imgs = new_imgs.reshape(tuple([new_imgs.shape[0]] + list(entrada_x.shape[1:])))
  new_label = np.array(new_label)
  all_CTs = np.append(entrada_x, new_imgs, axis = 0)
  all_labels = np.append(entrada_y, new_label, axis = 0)
  all_CTs, all_labels = shuffle(all_CTs, all_labels, random_state=42) #shuffle/mistura dos dados
  all_CTs = all_CTs.reshape(all_CTs.shape[0], 20, img_size, img_size, 1) #fazendo channels_last
  return all_CTs, all_labels

def rotaciona(img, label, n_dados, new_imgs, new_label, angulo = 360, fill_mode='nearest'):
    for rotacoes in range(n_dados):
      image = imggen.random_rotation(img, 360, fill_mode='nearest')
      new_imgs.append(np.array(image))
      new_label.append(label)

def data_aug_v2(entrada_x, entrada_y, n_dados, classe_aug = -1):
  new_imgs = []
  new_label = []
  for paciente in range(len(entrada_x)):
    img = entrada_x[paciente].reshape(20,img_size,img_size)
    label = entrada_y[paciente]
    if classe_aug != -1:
        if label == classe_aug:
            rotaciona(img, label, n_dados, new_imgs, new_label, 360, 'nearest')
    else:
        rotaciona(img, label, n_dados, new_imgs, new_label, 360, 'nearest')
  new_imgs = np.array(new_imgs)
  new_imgs = new_imgs.reshape(tuple([new_imgs.shape[0]] + list(entrada_x.shape[1:])))
  new_label = np.array(new_label)
  all_CTs = np.append(entrada_x, new_imgs, axis = 0)
  all_labels = np.append(entrada_y, new_label, axis = 0)
  all_CTs, all_labels = shuffle(all_CTs, all_labels, random_state=42) #shuffle/mistura dos dados
  return all_CTs, all_labels

def model(n_neuronios1, n_neuronios2, drop, regul):
    classificador = Sequential() 
    
    classificador.add(Conv3D(filters=16, kernel_size =(3,16,16), strides=(1, 4, 4), input_shape = (20,img_size,img_size,1), activation = 'relu', padding='same')) 
    classificador.add(LayerNormalization())
    classificador.add(MaxPooling3D(pool_size = (2,2,2), strides=(2, 2, 2))) 
    
    classificador.add(Conv3D(filters=8, kernel_size =(3,16,16), strides=(1, 4, 4), activation = 'relu', padding='same'))  
    classificador.add(LayerNormalization())   
    classificador.add(MaxPooling3D(pool_size = (2,2,2), strides=(2, 2, 2))) 
    
    classificador.add(Flatten())
  
    classificador.add(Dense(units = n_neuronios1, activation = 'relu', kernel_regularizer=regularizers.l2(regul))) 
    classificador.add(Dropout(drop))
  
    classificador.add(Dense(units = n_neuronios2, activation = 'relu', kernel_regularizer=regularizers.l2(regul)))
    classificador.add(Dropout(drop))
  
    classificador.add(Dense(units = 1, activation = 'sigmoid')) #2 opcoes de saida 1 ou 0
  
    opt = Adam(learning_rate=0.00001) #learning rate de 10-5
    classificador.compile(loss = 'binary_crossentropy', optimizer = opt,
                        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()],
                        weighted_metrics=['accuracy']
                        )  
    return classificador

img_size = 128
all_CT = np.load(f'C:\\Users\\Larissa\\.spyder-py3\\all_CT_20_{img_size}x{img_size}.npy')
all_label = np.load(f'C:\\Users\\Larissa\\.spyder-py3\\all_label_20_{img_size}x{img_size}.npy')

print(pd.Series(list(all_label)).value_counts())

seed = 5
kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = seed)
resultados = []
str_today = datetime.date.today().strftime('%Y-%m-%d')

n_neuronios1 = 8
n_neuronios2 = 8
drop = 0.6
regularizacao = 0.0001
epocas = 400
n_dados1 = 20 #numero que imagens geradas a partir de cada original para classe menor (1)
n_dados0 = 10 #numero que imagens geradas a partir de cada original para classe maior (0)
paciencia = 50 #por quantas epocas vamos esperar ate parar a execucao
delta = 0.01 #qual min mudanca que consideramos pra dizer que melhorou a metrica

acc_balanceada = []
i = 1
for indice_treino, indice_val in kfold.split(all_CT, all_label):
  start_time = time()
  print()
  print(f'Validacao {i}')
  print(indice_val)
  temp = pd.DataFrame(all_label, columns = ['label'])
  temp1 = pd.DataFrame(all_label[indice_treino], columns = ['label'])
  temp2 = pd.DataFrame(all_label[indice_val], columns = ['label'])
  print(f"doentes no total: {temp['label'].sum()} ({round(temp['label'].sum()/temp['label'].count()*100, 2)})%")
  print(f"doentes no treino: {temp1['label'].sum()} ({round(temp1['label'].sum()/temp1['label'].count()*100, 2)})%")
  print(f"doentes na val: {temp2['label'].sum()} ({round(temp2['label'].sum()/temp2['label'].count()*100, 2)})%")
  print()
  #data augmentation dos dados
  train_aug_CTs, train_aug_labels = data_aug_v2(all_CT[indice_treino], all_CT[indice_treino], n_dados1, 1)
  train_aug_CTs, train_aug_labels = data_aug_v2(train_aug_CTs, train_aug_labels, n_dados0, 0)

  #channels last
  all_CT = all_CT.reshape(all_CT.shape[0], 20, img_size, img_size, 1)
  train_aug_CTs = train_aug_CTs.reshape(train_aug_CTs.shape[0], 20, img_size, img_size, 1)
  print("data aug realizado")
  
  X_val = all_CT[indice_val]
  y_val = all_label[indice_val]

  modelo = model(n_neuronios1, n_neuronios2, drop, regularizacao)
  # Salva o modelo focando em maximizar o auc da validacao.
#  checkpoint = ModelCheckpoint(f"C:\\Users\\Larissa\\.spyder-py3\\lari_model_{i}_{str_today}.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
  # Pausa o modelo com base no erro da validação e salva os melhores pesos.
  early = EarlyStopping(monitor='val_loss', min_delta=delta, patience=paciencia, verbose=1, mode='min', restore_best_weights = True)

  peso_classes = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(train_aug_labels), y = np.ravel(train_aug_labels))  #calcula peso pra equilibrar proporcao entre classes
  history = modelo.fit(train_aug_CTs, train_aug_labels,
                    batch_size = 36, epochs = epocas, class_weight = {0: peso_classes[0], 1: peso_classes[1]}, 
                    validation_data = (X_val, y_val), callbacks=[early])

  # vamos avaliar as metricas com base na validação
  y_pred = (modelo.predict(X_val) > 0.5).astype("int32")
  acc_balanceada.append(balanced_accuracy_score(y_val, y_pred))

  metricas = modelo.evaluate(X_val, y_val)
  resultados.append(metricas)
  print(metricas)

  backend.clear_session()

  plt.plot(history.history['accuracy'], 'b')
  plt.plot(history.history['val_accuracy'], 'g')
  plt.xlabel('Epochs')
  plt.ylabel('accuracy')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

  plt.plot(history.history['loss'], 'b')
  plt.plot(history.history['val_loss'], 'g')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend(['train', 'validation'], loc='upper right')
  plt.show()
  
  end_time = time()
  tempo_total = end_time - start_time

  print(f'tempo da validacao {i}: {tempo_total/60} minutos')
  
  print(classification_report(y_val, y_pred))

  print(f'balanced accuracy: {balanced_accuracy_score(y_val, y_pred)}')

  print(f'Total tests: {len(y_val)}')
  
  cm = confusion_matrix(y_val, y_pred)
  plt.figure(figsize = (5,5))
  sn.heatmap(cm, annot = True)
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()
  
  linhas = int(np.ceil(len(X_val)/5))
  fig, ax = plt.subplots(nrows=linhas, ncols=5, figsize=(15,10))
  fig.subplots_adjust(hspace=0.6, wspace=0.5)
  j = 0
  for linha in ax:
    for coluna in linha:
      if j == X_val.shape[0]:
        break
      coluna.imshow(X_val.reshape(X_val.shape[0], 1, 20, img_size, img_size)[j][0][10], cmap = plt.cm.bone)
      coluna.set_title(f'Previsto:{y_pred[j]}, Correto:{y_val[j]}')
      j += 1
      if j == len(X_val):
          break
  plt.show()
  
  i += 1
  
loss = []
acuracia = []
auc = []
precisao = []
recall = []
sensib = []
espec = []
acuracia_balanceada = []

for i in range(len(resultados)):
  VP = resultados[i][5]
  VN = resultados[i][6]
  FP = resultados[i][7]
  FN = resultados[i][8]
  loss.append(resultados[i][0])
  acuracia.append(resultados[i][1])
  auc.append(resultados[i][2])
  precisao.append(resultados[i][3])
  recall.append(resultados[i][4])
  sensib.append(VP/(VP+FN))
  espec.append(VN/(VN+FP))
  acuracia_balanceada.append(resultados[i][9])
  
print(f'Loss medio: {np.mean(loss)}')
print(f'Acuracia media: {round(np.mean(acuracia)*100, 2)}%')
print(f'AUC medio: {round(np.mean(auc)*100, 2)}%')
print(f'Precisao media: {round(np.mean(precisao)*100, 2)}%')
print(f'Recall medio: {round(np.mean(recall)*100, 2)}%')
print(f'Sensibilidade media: {round(np.mean(sensib)*100, 2)}%')
print(f'Especificidade media: {round(np.mean(espec)*100, 2)}%')
print(f'Acuracia balanceada media: {round(np.mean(acuracia_balanceada)*100, 2)}%')
print(f'acur. balanceada scikit: {round(np.mean(acc_balanceada)*100, 2)}%')

print()
print(f'loss: {loss}')
print(f'acuracia: {acuracia}')
print(f'AUC: {auc}')
print(f'precisao: {precisao}')
print(f'recall: {recall}')
print(f'sensibilidade: {sensib}')
print(f'especificidade: {espec}')
print(f'acur. balanceada: {acuracia_balanceada}')
print(f'acur. balanceada do scikit: {acc_balanceada}')