import torch
from pet_dataset import PetDataset
import albumentations as A
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as models
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import os 


# Paths
data_root = "/home/ali/data/petfinder_pawpularity_score/"

# Hyper params
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_BINS = 10
EPOCHS = 20
BATCH_SIZE = 32

# Reproducability
SEED = 1984
torch.backends.cudnn.deterministic = True
# random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


# Augmentation pipelines
train_augments = A.Compose([
    A.LongestMaxSize(max_size=224, interpolation=1),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0,0,0)),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
#     A.ToFloatV2(),
])
test_augments = A.Compose([
    A.LongestMaxSize(max_size=224, interpolation=1),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0,0,0)),
    A.Normalize(),
#     A.ToFloatV2(),
])

# Split validation set from train
orig_train_df = pd.read_csv(os.path.join(data_root, "train.csv"))
train_df = orig_train_df.sample(frac=0.8, random_state=SEED) 
val_df = orig_train_df.drop(train_df.index)

train_dataset = PetDataset(data_root + "train/",
                        train_df,
                        augment_fn=train_augments,
                        num_bins=NUM_BINS,
                        as_tensor=True)
val_dataset = PetDataset(data_root + "train/",
                        val_df,
                        augment_fn=test_augments,
                        num_bins=NUM_BINS,
                        as_tensor=True)

train_loader = DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=6)

val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=6)
# images are NHWC


model = models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_BINS)
model.to(device=DEVICE)



loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.001, 
                            momentum=0.9, 
                            dampening=0, 
                            weight_decay=1e-5, 
                            nesterov=False)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=0.75,
                                                         patience=3,
                                                         mode='min',
                                                         verbose=True)




def train(model, dataloader, criterion, optimizer, num_epochs, epoch, debug=False):
    model.train()
    running_loss = 0.
    for train_epoch in range(num_epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, sample in pbar:
            pbar.set_description(f"Epoch {epoch+1} | train loss = {round(running_loss / (i + 1), 3)}",
                                refresh=True)
            inputs = sample['image'].to(device=DEVICE)
            labels = sample['label'].to(device=DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            # print(f"train loss: {running_loss / (i+1)}")            
            
            if debug and i > 2:
                break

    train_loss = running_loss / (len(dataloader) * num_epochs)
    return train_loss


def validate(model, dataloader, criterion, total_set_size, debug=False):
    model.eval()
    running_loss = 0.
    running_correct = 0
    

    for i, sample in enumerate(dataloader):
        inputs = sample['image'].to(device=DEVICE)
        labels = sample['label'].to(device=DEVICE)

        with torch.no_grad():
            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Update loss and accuracy
            running_loss += loss.item()
       
            # softmax across logits
            softmax = nn.Softmax(dim=1)
            preds = softmax(outputs)
            # argmax so both tensors are no longer one-hot
            preds_argmax = preds.argmax(dim=-1)
            labels_argmax = labels.argmax(dim=-1)
            # add correct preds to running total
            running_correct += (preds_argmax == labels_argmax).sum().item()
        
        if debug and i > 2:
            break

    val_loss = running_loss / len(dataloader)
    # use total dataset size so we can have uneven number of elements in last batch
    val_acc = running_correct / total_set_size
    
    
    print(f"val loss: {val_loss}")     
    print(f"val acc: {val_acc}")     


    return val_loss, val_acc


best_val_loss = np.inf
now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")
# os.makedirs(f"./logs/{dt_string}/", exist_ok=True)
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, loss, optimizer, num_epochs=1, epoch=epoch)

    val_loss, val_acc = validate(model, val_loader, loss, total_set_size=len(val_dataset))
    lr_scheduler.step(val_loss)

    # if val_loss < best_val_loss:
    #     print(f"Best val loss: {val_loss}")
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), f"./logs/{dt_string}/best.pt")