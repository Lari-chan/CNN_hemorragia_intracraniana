# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:40:30 2021

@author: Larissa
"""

import pydicom as dicom
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import cv2
import math 
from sklearn.preprocessing import MinMaxScaler

def load_scan(path):
    slices = [dicom.dcmread(path + '/' + s) for s in               
              os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))  #InstanceNumber
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    # The intercept is usually -1024, so air is approximately 0 -> no nosso caso realmente eh -1024
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def corrigir_nome(nome):
    nome_certo = nome.split(' ')[0]
    return nome_certo

def retirar_tracos(dataframe):
    indexes = np.array(dataframe.index) #transforma os indices do dataframe em um array
    for i in range(len(indexes)): #pra cada um dos indices
        indexes[i] = indexes[i].replace('-','') #retira todos os - do indice
    return indexes

def chunks(l, nr_of_chunks):
    i = 0
    chunk_size = len(l) / nr_of_chunks
    while chunk_size * i < len(l):
        yield l[math.floor(chunk_size * i):math.floor(chunk_size * (i + 1))]
        i += 1

def mean(l):
    return sum(l)/len(l)

def decisao_unanime(paciente):
    if (labels_orig['R1:ICH'][paciente] == labels_orig['R2:ICH'][paciente] == labels_orig['R3:ICH'][paciente]):
        return True
    else: return False

def process_data(patient, labels_df, img_size = 128, qt_slices = 20, visualize = False):
    nome = corrigir_nome(patient)
    #print(f'Paciente {nome}')
    label = labels_df.at[nome, 'edema']
    path = data_dir + patient
    slices = load_scan(path)
    n_slices = len(slices)
    #print(f'Num de slices: {n_slices}')
    #print(f'Tamanho dos slices: {slices[0].pixel_array.shape})
    avg_slices = []
    new_slices = [cv2.resize(np.array(img.pixel_array), (img_size,img_size)) for img in slices]
    for slice_chunk in chunks(new_slices, qt_slices):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        avg_slices.append(slice_chunk)   
    if visualize:
        fig = plt.figure()
        for n, img in enumerate(avg_slices):
            y = fig.add_subplot((qt_slices/5),5,n+1)
            y.imshow(img, cmap=plt.cm.bone) #pra colorido tirar o cmap
        plt.show    
    return np.array(avg_slices), label, n_slices

def normalizacao_slices(a):
    scaler = MinMaxScaler(feature_range=(0, 1)) #prepara padronizacao
    for i in range(a.shape[0]):
        b = scaler.fit_transform(a[i])
        a[i] = b
    return a


# Salvar lista de pacientes com base nas pastas de arquivos (patients) e salvar diagnosticos deles com base no csv (labels_df) 
data_dir = 'C:/Users/Larissa/.spyder-py3/CT_dados/patients/'
patients = os.listdir(data_dir)
quant_pacientes = len(patients)
labels_orig = pd.read_csv('C:/Users/Larissa/.spyder-py3/CT_dados/reads.csv', index_col = 'name')
labels_df = pd.Series()
for paciente in labels_orig.index:
    if decisao_unanime(paciente):
        labels_df = labels_df.append(pd.Series({paciente:labels_orig['R1:ICH'][paciente]}))
labels_df = pd.DataFrame(labels_df)
labels_df['novo_indice'] = retirar_tracos(labels_df) #cria nova coluna com esses novos indices
labels_df.set_index('novo_indice',inplace = True) #substitui os index por essa coluna. sem o inplace ele nao substitui de verdade
labels_df.columns = ['edema']

# Pre-processamento das imagens
    # Variaveis para redimensionamento e padronizacao das imagens -> CT com (qt_slices) imagens quadradas de lado (img_size)
tam_img = 128
quant_slices = 20 #multiplos de 5
qt_original_slices = []

all_CT = []
all_label = []

for num, patient in enumerate(patients):    
    print(f'Processando paciente {num+1}')
    if corrigir_nome(patient) in labels_df.index:
        print(patient)
        CT, classe, num_slices_orig = process_data(patient, labels_df, img_size = tam_img, qt_slices = quant_slices)
        CT_norm = normalizacao_slices(CT)
        # CT_norm = normaliza(CT)
        qt_original_slices.append(num_slices_orig)
        all_CT.append([CT_norm])
        all_label.append([classe])
np.save('all_CT_{}_{}x{}_v4.npy'.format(quant_slices, tam_img, tam_img), all_CT)
np.save('all_label_{}_{}x{}_v4.npy'.format(quant_slices, tam_img, tam_img), all_label)
print(f'Numero de pacientes: {len(all_CT)}')
print(f'Quantidade de slices: {min(qt_original_slices)} a {max(qt_original_slices)} \nTamanho: 512x512')
print(f'Padronizado para {quant_slices} slices de tamanho {tam_img}x{tam_img}')
