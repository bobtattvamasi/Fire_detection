import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F

from PIL import Image


labels = ['Unlabeled','Fire']

# Теперь обернем все в кастомный датасет для удобной работы в PyTorch.
class FireDataset(Dataset):
    def __init__(self, data_info):
        # Подаем наш подготовленный датафрейм
        self.data_info = data_info
        
        # Разделяем датафрейм на rgb картинки 
        self.image_arr = self.data_info.iloc[:,0]
        # и на сегментированные картинки
        self.label_arr = self.data_info.iloc[:,1]
        
        # Количество пар картинка-сегментация
        self.data_len = len(self.data_info.index)
        
        # convert str names to class values on masks
        # Конвертируем стринговые имена в значения класса на маске
        self.class_values = [labels.index(cls) for cls in labels]
        
    def __getitem__(self, index):
        # Читаем картинку и сразу же представляем ее в виде numpy-массива 
        # размера 600х800 float-значний
        img = Image.open(self.image_arr[index])
        # Преобразовываем к размеру 256х256
        img = img.resize((256,256))
        img = np.asarray(img).astype('float')
        # Нормализуем изображение в значениях [0,1]
        img = torch.as_tensor(img)/255    
        # Количество каналов ставим на первый план - этого хочет pytorch
        img = img.permute(2,0,1).float()
        
        # Считываем нашу маску
        mask = np.asarray(plt.imread(self.label_arr[index]))[:,:,0]*255

        # Здесь мы создаем список бинарных масок из нашей одной общей маски 
        masks = [(mask == v) for v in self.class_values]
        # Стекаем все вместе в один многомерный тензор масок
        mask = np.stack(masks, axis=-1).astype('float')
        #  Приводим к типу тензора
        mask = torch.as_tensor(mask)
        # Размерность каналов на передний план
        mask = mask.permute(2,0,1)
        
        # делаем ресайз маски на 256х256
        # Для этого используем функцию interpolate
        ### Что бы ресайзить и высоту и ширину картинки, нужно перед interpolate
        ### пороизвести unsqueeze над тензором, и squeeze после.
        # unsqueeze - меняет размерность img c (256, 256, 3) -> (1, 256, 256, 3),
        mask = mask.unsqueeze(0)
        mask = F.interpolate(input=mask, size=256, mode='nearest')
        mask=mask.squeeze(0).squeeze(0)
        
        
        return (img, mask)

    def __len__(self):
        return self.data_len

class getDataLoader:
    '''
        Класс для загрузки наших данных куда надо.
    '''
    def __init__(self):
        self.folder_name = 'VF_dataset'
        self.image_folder_name = 'img'
        self.masks_folder_name = 'masks_machine'

    def to_df(self):
        cameraRGB = []
        cameraSeg = []
        for root, dirs, files in os.walk(self.folder_name):
            for name in files:
                f = os.path.join(root, name)
                if self.image_folder_name in f:
                    cameraRGB.append(f)
                elif self.masks_folder_name in f:
                    cameraSeg.append(f)
                else:
                    break

        # Теперь завернем эти два списка в DataFrame
        df = pd.DataFrame({'cameraRGB': cameraRGB, 'cameraSeg': cameraSeg})
        # Отсортируем  датафрейм по значениям
        df.sort_values(by='cameraRGB',inplace=True)
        # Используем функцию,
        # лагодаря которой индексация значений 
        # будет начинаться с 0.
        df.reset_index(drop=True, inplace=True)
        return df

    def return_data(self):
        # Затем разделим наш датасет на тренировочную и тестовую выборки. 
        # 70 % в тренировочную выборку, 30 - в тестовую
        X_train, X_test = train_test_split(self.to_df(),test_size=0.3)

        # Упорядочиваем индексацию
        X_train.reset_index(drop=True,inplace=True)
        X_test.reset_index(drop=True,inplace=True)

        # Оборачиваем каждую выборку в наш кастомный датасет
        train_data = FireDataset(X_train)
        test_data = FireDataset(X_test)

        # И теперь уже обернем то, что получилось в известные нам в pytorch даталоадеры
        train_data_loader = DataLoader(train_data,batch_size=8,shuffle=True)
        test_data_loader = DataLoader(test_data,batch_size=5,shuffle=False)

        return train_data_loader, test_data_loader
