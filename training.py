
# %%
# system level modules for handling files and file structures
import os
import tarfile
import copy

# scipy ecosystem imports for numerics, data handling and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# pytorch and helper modules
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import JaccardIndex, F1Score, Dice

# utils
from tqdm import tqdm

# rasterio for reading in satellite image data
import rasterio as rio

import itertools

from unet import UNet

data_base_path = 'C:/ImportantFolders/Documents/BachelorThesis/data_for_python'

 # define ESA WorldCover colormap
COLOR_CATEGORIES = [
        (0, 100, 0),
        (255, 187, 34),
        (255, 255, 76),
        (240, 150, 255),
        (250, 0, 0),
        (180, 180, 180),
        (240, 240, 240),
        (0, 100, 200),
        (0, 150, 160),
        (0, 207, 117),
        (250, 230, 160)]
cmap_all = mpl.colors.ListedColormap(np.array(COLOR_CATEGORIES)/255.)

def collate_fn(batch):
        return {'s2': torch.stack([x['s2'] for x in batch]),
                'lulc': torch.stack([x['lulc'] for x in batch])}

class BENGE(Dataset):
        """A dataset class implementing the Sentinel-1, Sentinel-2 and ESAWorldCover data modalities."""
        def __init__(self, 
                    data_dir=data_base_path, 
                    split='train',
                    s2_bands=[2, 3, 4, 8]):
            """Dataset class constructor

            keyword arguments:
            data_dir -- string containing the path to the base directory of ben-ge dataset, default: ben-ge-800 directory
            split    -- string, describes the split to be instantiated, either `train`, `val` or `test`
            s2_bands -- list of Sentinel-2 bands to be extracted, default: all bands

            returns:
            BENGE object
            """
            super(BENGE, self).__init__()

            # store some definitions
            self.s2_bands = s2_bands
            self.data_dir = data_dir

            # read in relevant data files and definitions
            self.split = split   # 80/10/10 split -> in eigene Verzeichnisse (train/, val/, test/)
            self.imgfiles = sorted(os.listdir(os.path.join(self.data_dir, self.split)))  # images
        

        def __getitem__(self, idx):
            """Return sample `idx` as dictionary from the dataset."""
            imgfile = self.imgfiles[idx]
            # print("imgfile path: " + imgfile)
            lblfile = 'esaworldcover_'+imgfile
            
            lblfile = f"esaworldcover_{os.path.splitext(imgfile)[0]}.tif"
            # imgfile und lblfile einlesen

            ### IMG UND LBL WERDEN HIER NIE AUFGERUFEN. WARUM? #######
            # with rio.open(self.data_dir + "/" + self.split + "/" + imgfile) as sentinelimage:    
                # img = sentinelimage.shape   
                # img = ... # shape: [4, 120, 120]
        
            # with rio.open(self.data_dir + "/"+ self.split + "/" + lblfile) as labelimage:
                # lbl = labelimage.shape
                # lbl = ...   # shape: [120, 120]
            
            # # retrieve Sentinel-2 data
            ################# ORIGINALER CODE ###############################################################
            # s2 = np.empty((4, 120, 120))
            # for i, band in enumerate(self.s2_bands):
                # with rio.open(f"{self.data_dir}/sentinel-2/{patch_id}/{patch_id}_B0{band}.tif") as dataset:
            #         data = dataset.read(1)
            #     if i < 3:
            #         s2[i,:,:] = data/3000  # normalize Sentinel-2 data
            #     else:
            #         s2[i,:,:] = data/6000  # normalize Sentinel-2 data
            ###################### ORIGINALER CODE ENDE ###################################################### 

            s2 = np.empty((4, 120, 120))
            # print("Self.s2_bands: ")
            # print(self.s2_bands)
            # print("S2: ")
            # print(s2)
            for i, band in enumerate(self.s2_bands):
                with rio.open(f"{self.data_dir}/{self.split}/{idx+1}.tif") as dataset:
                    data = dataset.read(1)
                    # print("Data: ", data)
                if i < 3:
                    s2[i,:,:] = data/3000  # normalize Sentinel-2 data
                else:
                    s2[i,:,:] = data/6000  # normalize Sentinel-2 data    

            ################################ ORIGINALER CODE ##################################################
            # extract lulc data
            # with rio.open(f"{self.data_dir}/esaworldcover/{patch_id}_esaworldcover.tif") as dataset:
            #     ewc_data = dataset.read(1)
            # ewc_mask = ewc_data.astype(float)   
            # ewc_mask[ewc_mask == 100] = 110  # fix some irregular class labels
            # ewc_mask[ewc_mask == 95] = 100   # fix some irregular class labels
            # ewc_mask = ewc_mask / 10 - 1 # transform to scale [0, 11]
            ############################## ORIGINALER CODE ENDE ##################################

            # extract lulc data
            with rio.open(f"{self.data_dir}/{self.split}/esaworldcover_{idx+1}.tif") as dataset:
                ewc_data = dataset.read(1)
            ewc_mask = ewc_data.astype(float)   
            ewc_mask[ewc_mask == 100] = 110  # fix some irregular class labels
            ewc_mask[ewc_mask == 95] = 100   # fix some irregular class labels
            ewc_mask = ewc_mask / 10 - 1 # transform to scale [0, 11]

            # create sample dictionary containing all the data
            sample = {
                #"patch_id": patch_id,  # Sentinel-2 id of this patch
                "s2": torch.from_numpy(s2).float(),  # Sentine;-2 data [4, 120, 120]
                "lulc": torch.from_numpy(ewc_mask).long(),  # ESA WorldCover lulc classes per pixel [120, 120]
                }

            return sample

        ##### ORIGINALER CODE #######
        # def __len__(self):
        #     """Return length of this dataset."""
        #     return self.meta.shape[0]
        ##### ORIGINALER CODE ENDE ######

        def __len__(self):
            """Return length of this dataset."""
            return int(len(self.imgfiles)/2)
            # return 39376

        def display(self, idx, pred=None):
            """Method to display a data sample, consisting of the Sentinel-2 image and lulc map, and potentially a corresponding prediction.
            
            positional arguments:
            idx -- sample index
            
            keyword arguments:
            pred -- prediction tensor
            """

            # retrieve sample
            sample = self[idx]

            if pred is None:
                f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
            else:
                f, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))

            # display Sentinel-2 image
            img_rgb = np.dstack(sample['s2'][0:3].numpy()[::-1])  # extract RGB, reorder, and perform a deep stack (shape: 120, 120, 3)
            img_rgb = np.clip((img_rgb-np.percentile(img_rgb, 1))/(np.percentile(img_rgb, 99)-np.percentile(img_rgb, 1)), 0, 1)
            ax[0].imshow(img_rgb)
            ax[0].set_title('Sentinel-2')
            ax[0].axis('off')

            # display lulc map
            ax[1].imshow(sample['lulc'], cmap=cmap_all, vmin=0, vmax=11, interpolation='nearest')
            ax[1].set_title('LULC')
            ax[1].axis('off')

            # display prediction, if available
            if pred is not None:
                ax[2].imshow(pred, cmap=cmap_all, vmin=0, vmax=11, interpolation='nearest')
                ax[2].set_title('Prediction')
                ax[2].axis('off')



def main():

    np.random.seed(42)     # sets the seed value in Numpy
    torch.manual_seed(42)  # sets the seed value in Pytorch\
    torch.cuda.manual_seed(42)

    # class label names
    ewc_label_names = ["tree_cover", "shrubland", "grassland", "cropland", "built-up",
                    "bare/sparse_vegetation", "snow_and_ice","permanent_water_bodies",
                    "herbaceous_wetland", "mangroves","moss_and_lichen"]

    train_data = BENGE(split='train')
    # train_data.data_dir = train_data.data_dir + "/train"
    val_data = BENGE(split='val')
    # val_data.data_dir = val_data.data_dir + "/val"
    test_data = BENGE(split='test')
    # test_data.data_dir = test_data.data_dir + "/test"


    print("S2 bands: " + str(train_data.s2_bands))

    print(train_data.data_dir)
    print("train_data[0]: ")
    # print(train_data[0])
    # train_data.display(0)

    ######### stop hier

    train_batchsize = 64
    eval_batchsize = 128

    # print(train_data[0])
    train_dataloader = DataLoader(train_data, batch_size=train_batchsize, num_workers=12, pin_memory=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=eval_batchsize, num_workers=12, pin_memory=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=eval_batchsize, num_workers=4, pin_memory=True, collate_fn=collate_fn)


    model = UNet(n_channels=4, n_classes=12)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Running with CUDA")


    # we will use the cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # we will use the Adam optimizer
    learning_rate = 0.001
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)

    # we instantiate the iou metric
    iou = JaccardIndex(task="multiclass", num_classes=12).to(device)
    f1 = F1Score(task="multiclass", num_classes=12).to(device)
    dice = Dice(num_classes=12, multiclass=True).to(device).to(device)

    saved_model_path = './final_run/run_49.pth'  # Replace with your actual file name
    model.load_state_dict(torch.load(saved_model_path, map_location=device)) 
    model.to(device)
    criterion.to(device)
    iou.to(device)
    f1.to(device)
    dice.to(device)

    epochs = 0

    train_losses_epochs = []
    val_losses_epochs = []
    train_ious_epochs = []
    val_ious_epochs = []
    train_f1_epochs = []
    val_f1_epochs = []
    train_dice_epochs = []
    val_dice_epochs = []

    for ep in range(epochs):
        
        train_losses = []
        val_losses = []
        train_ious = []
        val_ious = []
        train_f1 = []
        val_f1 = []
        train_dice = []
        val_dice = []

        # we perform training for one epoch
        
        model.train()   # it is very important to put your model into training mode!
        pbar = tqdm(train_dataloader)
        dataloader_iter = iter(train_dataloader)
        # for samples in tqdm(train_dataloader):
        
        for _ in tqdm(range(len(train_dataloader))):
                try:
                    samples = next(dataloader_iter)  # Fetch the next batch manually
                    # we extract the input data (Sentinel-2)
                    x = samples['s2'].to(device)
                    # print(x)

                    # now we extract the target (lulc class) and move it to the gpu
                    y = samples['lulc'].to(device)
                
                    # we make a prediction with our model
                    output = model(x)
                    
                    # we reset the graph gradients
                    model.zero_grad()

                    # we determine the classification loss
                    loss_train = criterion(output, y)

                    # we run a backward pass to comput the gradients
                    loss_train.backward()

                    # we update the network paramaters
                    opt.step()

                    # we write the mini-batch loss and accuracy into the corresponding lists
                    train_losses.append(loss_train.detach().cpu())
                    train_ious.append(iou(torch.argmax(output, dim=1), y).detach().cpu())
                    train_f1.append(f1(torch.argmax(output, dim=1), y).detach().cpu())
                    train_dice.append(dice(torch.argmax(output, dim=1), y).detach().cpu())
                    
                    pbar.set_description("loss={:.4f}, iou={:.4f}, f1={:.4f}".format(train_losses[-1], train_ious[-1], train_f1[-1]))
                except rio.errors.RasterioIOError:
                    continue
       
        # we evaluate the current state of the model on the validation dataset
        model.eval()   # it is very important to put your model into evaluation mode!
        with torch.no_grad():
            for _ in tqdm(range(len(val_dataloader))):
                try:
                    # we extract the input data (Sentinel-2)
                    x = samples['s2'].to(device)

                    # now we extract the target (lulc class) and move it to the gpu
                    y = samples['lulc'].to(device)
                    
                    # we make a prediction with our model
                    output = model(x)

                    # we determine the classification loss
                    loss_val = criterion(output, y)

                    # we write the mini-batch loss and accuracy into the corresponding lists
                    val_losses.append(loss_val.detach().cpu())
                    val_ious.append(iou(torch.argmax(output, dim=1), y).detach().cpu())
                    val_f1.append(f1(torch.argmax(output, dim=1), y).detach().cpu())
                    val_dice.append(dice(torch.argmax(output, dim=1), y).detach().cpu())
                except rio.errors.rasterioIOerror:
                    continue
        train_losses_epochs.append(np.mean(train_losses))
        train_ious_epochs.append(np.mean(train_ious))
        train_f1_epochs.append(np.mean(train_f1))
        train_dice_epochs.append(np.mean(train_dice))

        val_losses_epochs.append(np.mean(val_losses))
        val_ious_epochs.append(np.mean(val_ious))
        val_f1_epochs.append(np.mean(val_f1))
        val_dice_epochs.append(np.mean(val_dice))
        
        print("epoch {}: train: loss={:.4f}, iou={:.4f}, f1={:.4f}, dice={:.4f}; val: loss={:.4f}, iou={:.4f}, f1={:.4f}, dice={:4f}".format(
            ep, train_losses_epochs[-1], train_ious_epochs[-1], train_f1_epochs[-1], train_dice_epochs[-1],
            val_losses_epochs[-1], val_ious_epochs[-1], val_f1_epochs[-1], val_dice_epochs[-1]))

        with open('train.log', 'a') as outf:
            outf.write('{},{},{},{},{},{},{},{},{}\n'.format(
                ep, train_losses_epochs[-1], train_ious_epochs[-1], train_f1_epochs[-1], train_dice_epochs[-1],
                val_losses_epochs[-1], val_ious_epochs[-1], val_f1_epochs[-1], val_dice_epochs[-1]))
        ep_name = ep
        torch.save(model.state_dict(), 'C:/ImportantFolders/Documents/BachelorThesis/Mommert_Trainingsskript_und_U-Net/final_run/run_{:02d}.pth'.format(ep_name)) 
        print("Done with training")
    
    dataloader_iter_display = iter(train_dataloader)
    for i in range(0,1):
        item_to_display = next(dataloader_iter_display)
    x = item_to_display['s2'].to(device)
    output = model(x)
    output_classes = torch.argmax(output, dim=1)
    output_classes = output_classes[0].cpu().numpy() 
    output_classes = np.squeeze(output_classes)
    plt.imshow(output_classes, cmap="tab20")  # Use 'tab20' colormap for multi-class segmentation
    plt.colorbar()  # Show colorbar to indicate different classes
    plt.title('Predicted Class Map')
    plt.show()  # Display the plot
if __name__ == '__main__':
    main()
