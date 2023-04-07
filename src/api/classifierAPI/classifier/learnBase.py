import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import Config
from datasets.customDataset import CustomDataset
from models.model4 import Network4
from datasetUtils import splitDataset
from classifierConfig import ClassifierConfig

BATCH_SIZE = 10  # size of batches in the dataset
CLASSES = ('0', '1', '2')  # class labels
config = ClassifierConfig(Config.getPath())
config.print()

print('Starting...')
if __name__ == '__main__':
    print('Starting...')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    net = Network4()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.getLearningRate(), momentum=config.getMomentum())

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataSet = CustomDataset(path=config.getDataPath(), classes=CLASSES,
                            img_dim=config.getImageSize(), transform=transform)

    dataSets = splitDataset(dataSet, test_size=config.getTestRatio())
    dataLoaders = {x: DataLoader(
        dataSets[x], BATCH_SIZE, shuffle=True, num_workers=config.getDataLoaderWorkers()) for x in ['train', 'test']}

    if config.getLoadModel():
        net.load_state_dict(torch.load(config.getModelPath()))
        print('Model loaded')
        
    # send network to device
    net.to(device)
    # set training mode
    net.train()

    for epoch in range(config.getEpochs()):  # loop over the dataset multiple times

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

    if config.getSaveModel():
        torch.save(net.state_dict(), config.getModelPath())
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
