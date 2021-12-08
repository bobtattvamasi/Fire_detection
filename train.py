import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import time
from torch.utils.tensorboard import SummaryWriter
import torch

from model import UNet
from dataLoader import getDataLoader

savePATH = '.saved_models/selfLastModel_dice_loss1.pth'

labels = ['Unlabeled','Fire']

train_data_loader, test_data_loader = getDataLoader().return_data()

#-----------------------
# У нас есть готовые данные и определенная сеть,
# которую мы хотим обучить. Пришло время построить базовый обучающий конвейер.
#------------------------

# Выберем устройство,на котором будем обучать и тестировать нашу модель:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Определим скорость обучения и количество эпох:
learning_rate = 0.001
epochs = 5

# Определим нашу модель
Umodel = UNet(num_classes=1).to(device)

# Под обучением мы понимаем скармливание целевой функции оптимизирующей функции. 
# Поэтому выберем оптимизирующую функцию и функцию потерь (целевая функция):
optimizer = torch.optim.Adam(Umodel.parameters())

# Определим количество шагов внутри одной эпохи:
total_steps = len(train_data_loader)
print(f"{epochs} epochs, {total_steps} total_steps per epoch")

# В качестве функции потерь воспользуемся дайс лоссом
# из библиотеки segmentation_models_pytorch
criterion = smp.utils.losses.DiceLoss()

# Для отслеживания процесса тренировки
train_writer = SummaryWriter()

# Запускаем сам процесс обучения:
epoch_losses = []
Umodel.train()
for epoch in range(epochs):
    time1 = time.time()
    epoch_loss = []
    for batch_idx, (data, labels) in enumerate(train_data_loader):
        
        data, labels = data.to(device), labels.to(device)        
        
        optimizer.zero_grad()

        outputs = Umodel(data)
        
        loss = criterion(outputs, labels)
        
        train_writer.add_scalar("Loss/batch", loss, batch_idx)
        
        
        loss.backward()
        optimizer.step()
        
        epoch_loss.append(loss.item())
        
        if batch_idx%200==0:
            print(f'batch index : {batch_idx} | loss : {loss.item()}')

    print(f'Epoch {epoch+1}, loss: ',np.mean(epoch_loss))
    time2 = time.time()
    print(f'Spend time for 1 epoch: {time2-time1} sec')
    
    train_writer.add_scalar("Loss/Epoch", np.mean(epoch_loss), epoch+1)
    
    epoch_losses.append(epoch_loss)

train_writer.flush()

train_writer.close()

torch.save(Umodel.state_dict(), savePATH)
