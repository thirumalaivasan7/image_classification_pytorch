from dataloader import inaturalist
from model import ResNet
from model import baseBlock
from model import bottleNeck
import torch.nn as nn
import torch.optim as optim
import os
import time
import torch
from torch.utils.data import DataLoader
from torchsummary import summary


checkpoint_dir = 'checkpoints'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)


#def get_model_summary(model, input_tensor_shape):
    #summary(model, input_tensor_shape)


def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return correct / total


def train(model, dataset, optimizer,criterion, device):
    '''
    Write the function to train the model for one epoch
    Feel free to use the accuracy function defined above as an extra metric to track
    '''
    # ------YOUR CODE HERE-----#
    running_loss = 0.0
    for data in dataset:
        images,labels = data[0].to(device),data[1].to(device)
        outputs = model(images)
        optimizer.zero_grad()
        loss = criterion(outputs,labels)
        running_loss +=loss
        loss.backward()
        optimizer.step()
    return running_loss

def eval(model, dataset, criterion, device):
    with torch.no_grad():
       total = 0.0
       correct = 0.0
       for data in dataset:
           images, labels = data[0].to(device), data[1].to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
       return correct / total

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Sections to Fill: Define Loss function, optimizer and model, Train and Eval functions and the training loop

def main():
    batch_size = 8
    epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet(bottleNeck, [3, 4, 6, 3])
    model = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
    trainset = inaturalist(root_dir='C:\studextra\choose_your_own\inaturalist_12K', mode='train')
    valset = inaturalist(root_dir='C:\studextra\choose_your_own\inaturalist_12K', mode='val')

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)


################################### DEFINE LOSS FUNCTION, MODEL AND OPTIMIZER ######################################
# USEFUL LINK: https://pytorch.org/docs/stable/nn.html#loss-functions
# ---Define the loss function to use, model object and the optimizer for training---#
################################### CREATE CHECKPOINT DIRECTORY ####################################################
# NOTE: If you are using Kaggle to train this, remove this section. Kaggle doesn't allow creating new directories.
#################################### HELPER FUNCTIONS ##############################################################
################################################### TRAINING #######################################################
# Get model Summary
    #get_model_summary(model, (3, 200, 200))

# Training and Validation

    best_valid_loss = float('inf')
    print(summary(model,(3,128,128)))
    for epoch in range(2):
        start_time = time.monotonic()
        epoch_loss = train(model,trainloader,optimizer,criterion,device)
        epoch_accuracy = eval(model,valloader,criterion,device)
        print("training_loss :{}  validation_accuracy:{}".format(epoch_loss,epoch_accuracy))
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print("\n\n\n TIME TAKEN FOR THE EPOCH: {} mins and {} seconds".format(epoch_mins, epoch_secs))

    print("OVERALL TRAINING COMPLETE")
    PATH = './RESNET2.pth'
    torch.save(model.state_dict(),PATH)
    '''''
    Insert code to train and evaluate the model (Hint: use the functions you previously made :P)
    Also save the weights of the model in the checkpoint directory
    '''
    PATH = './RESNET2.pth'
    model.load_state_dict(torch.load(PATH))
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print( correct / total)

if __name__ == '__main__':
       main()
