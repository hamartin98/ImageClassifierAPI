import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.customDataset import CustomDataset
from models.model4 import Network4
from datasetUtils import splitDataset

# TODO: Fix docker paths
# path to save to and load model from

MODEL_PATH = os.path.abspath('/data/models/model_latest.pth') # path to model for docker
#MODEL_PATH = os.path.relpath('data/models/model_latest.pth') # path to model

DATA_PATH = os.path.abspath('/data/images/first')  # path to data for docker
#DATA_PATH = os.path.relpath('data/images/first')  # path to data

BATCH_SIZE = 10  # size of batches in the dataset
#IMAGE_SIZE = (32, 32)  # size of images
IMAGE_SIZE = (62, 62)  # size of images
CLASSES = ('0', '1', '2')  # class labels
TEST_SIZE = 0.5  # size of test data set in percentages
SAVE_MODEL = False  # save model
LOAD_MODEL = True  # load model
EPOCHS = 10  # Number of epochs
DATALOADER_WORKERS = 2  # number of worker processes in the data loader
LEARNING_RATE = 0.001

print('Starting...')
if __name__ == '__main__':
    print('Starting...')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    net = Network4()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataSet = CustomDataset(path=DATA_PATH, classes=CLASSES,
                            img_dim=IMAGE_SIZE, transform=transform)

    dataSets = splitDataset(dataSet, test_size=TEST_SIZE)
    dataLoaders = {x: DataLoader(
        dataSets[x], BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKERS) for x in ['train', 'test']}

    if LOAD_MODEL:
        net.load_state_dict(torch.load(MODEL_PATH))
        print('Model loaded')
        
    # send network to device
    net.to(device)
    # set training mode
    net.train()

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataLoaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # send data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')

    if SAVE_MODEL:
        torch.save(net.state_dict(), MODEL_PATH)
        print('Model saved')

    print('Testing')
    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    # again no gradients needed
    with torch.no_grad():
        for data in dataLoaders['test']:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            labels = torch.flatten(labels)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
