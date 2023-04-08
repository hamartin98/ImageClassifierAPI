import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.customDataset import CustomDataset
from models.FirstNetwork import FirstNetwork
from datasetUtils import splitDataset
from classifierConfig import ClassifierConfig
from classificationMap import (
    BaseClassification)

from models.baseNetwork import BaseNetwork
from timeit import default_timer as timer
import datetime


class Teacher:
    def __init__(self, classification: BaseClassification, config: ClassifierConfig = None) -> None:
        self.classification = classification
        self.config = classification.getConfigutation()
        if config:
            self.config = config
        self.config.innerOverrideToType(self.classification.type)

        self.setupDevice()

        #self.network: BaseNetwork = FirstNetwork(self.classification.getClassNum())
        self.network: BaseNetwork = self.classification.getNetwork()
        self.classes: tuple = self.classification.getClassLabelsTuple()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.network.parameters(), lr=self.config.getLearningRate(), momentum=self.config.getMomentum())

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dataSet = CustomDataset(path=self.config.getDataPath(), classes=self.classes,
                                     imgDim=self.config.getImageSize(), transform=self.transform)

        self.dataSets = splitDataset(
            self.dataSet, testSize=self.config.getTestRatio())
        self.dataLoaders = {x: DataLoader(
            self.dataSets[x], self.config.getBatchSize(), shuffle=True, num_workers=self.config.getDataLoaderWorkers()) for x in ['train', 'test']}

        self.classification.loadModel()

        # send network to device
        self.network.to(self.device)

    def setupDevice(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device {self.device}')

    def train(self) -> None:
        # set training mode
        self.network.train()

        startTime = timer()
        correct = 0
        size = len(self.dataLoaders['train'].dataset)
        # loop over the dataset multiple times
        for epoch in range(self.config.getEpochs()):

            runningLoss = 0.0
            for i, data in enumerate(self.dataLoaders['train'], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # send data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                
                # print statistics
                runningLoss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    accuracy = 100 * correct / size
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {runningLoss / 200:.3f} accuracy: {accuracy / 200:.3f} %')
                    runningLoss = 0.0

        endTime = timer()
        trainingTime = endTime - startTime
        trainingStr = str(datetime.timedelta(seconds = round(trainingTime)))
        print('Finished Training')
        print(f'Training duration: {trainingStr}')

        self.classification.saveModel()

    def test(self) -> None:
        print('Testing')
        correctPred = {classname: 0 for classname in self.classes}
        totalPred = {classname: 0 for classname in self.classes}

        # again no gradients needed
        self.network.eval()
        with torch.no_grad():
            for data in self.dataLoaders['test']:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = torch.flatten(labels)
                outputs = self.network(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correctPred[self.classes[label]] += 1
                    totalPred[self.classes[label]] += 1

        # print accuracy for each class
        for className, correctCount in correctPred.items():
            accuracy = 100 * float(correctCount) / totalPred[className]
            print(f'Accuracy for class: {className:5s} is {accuracy:.1f} %')

    def trainAndTest(self) -> None:
        self.train()
        self.test()
