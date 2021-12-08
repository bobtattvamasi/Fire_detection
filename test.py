import matplotlib.pyplot as plt
import numpy as np
import torch

from model import UNet
from dataLoader import getDataLoader


labels = ['Unlabeled','Fire']
loadPATH = './saved_models/selfLastModel_dice_loss.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net=UNet(1).to(device)
net.load_state_dict(torch.load(loadPATH))

train_data_loader, test_data_loader = getDataLoader().return_data()

def get_orig(image):
    #image = images[0,:,:,:]
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = np.clip(image, 0, 1)
    return image

for i, data in enumerate(test_data_loader):
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    f, axarr = plt.subplots(1,3)

    for j in range(0,5):
        axarr[0].imshow(outputs.squeeze().detach().cpu().numpy()[j,:,:])
        axarr[0].set_title('Guessed labels')
        
        axarr[1].imshow(labels.squeeze().detach().cpu().numpy()[j,:,:])
        
        axarr[1].set_title('Ground truth labels')

        original = get_orig(images[j].cpu())
        axarr[2].imshow(original)
        axarr[2].set_title('Original Images')
        plt.show()
        plt.gcf().show()
        if i>5:
            break