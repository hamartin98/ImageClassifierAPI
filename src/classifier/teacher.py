from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .datasets.customDataset import CustomDataset
from .datasetUtils import splitDataset
from .classifierConfig import ClassifierConfig
from .classificationMap import (
    BaseClassification)

from .models.baseNetwork import BaseNetwork
from timeit import default_timer as timer
from .timeUtils import TimeUtils

from .activeTrainingInfo import ActiveTrainingInfo, TrainingStatus


class Teacher:
    def __init__(self, classification: BaseClassification, config: ClassifierConfig = None) -> None:
        self.classification = classification

        if config:
            self.classification.configure(config)
            self.config = self.classification.getConfigutation()

        self.config.innerOverrideToType(self.classification.type)

        self.setupDevice()

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

        self.lossData: List[float] = []
        self.accuracyData: List[float] = []
        self.epochTimes: List[str] = []

        # setup training info tracker
        ActiveTrainingInfo.reset()
        ActiveTrainingInfo.setConfig(self.config)
        ActiveTrainingInfo.setName('Training')
        ActiveTrainingInfo.setCurrentEpochs(0)
        ActiveTrainingInfo.setTotalEpochs(self.config.getEpochs())
        ActiveTrainingInfo.setStartTime(TimeUtils.getCurrentTime())
        ActiveTrainingInfo.setEndTime(0)
        ActiveTrainingInfo.setStatus(TrainingStatus.STARTED)

    def setupDevice(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device {self.device}')

    def test(self) -> None:
        print('Testing')
        correctPred = {classname: 0 for classname in self.classes}
        totalPred = {classname: 0 for classname in self.classes}

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
        accuracyPerClass = {}

        for className, correctCount in correctPred.items():
            accuracy = 100 * float(correctCount) / totalPred[className]
            accuracyPerClass[className] = accuracy
            print(f'Accuracy for class: {className:5s} is {accuracy:.1f} %')

        ActiveTrainingInfo.setAccuracyPerClass(accuracyPerClass)

    def train(self) -> None:
        try:
            ActiveTrainingInfo.setStatus(TrainingStatus.RUNNING)
            self.network.train()
            startTime = timer()

            epochs = self.config.getEpochs()
            for epoch in range(epochs):
                epochStartTime = timer()

                print(f'Epoch [ {epoch + 1} / {epochs} ]')
                ActiveTrainingInfo.stepCurrentEpochs(1)
                self.trainStep(self.dataLoaders['train'])
                self.validationStep(self.dataLoaders['test'])

                epochEndTime = timer()
                epochTime = TimeUtils.getTimeDiffStr(
                    epochStartTime, epochEndTime)
                self.epochTimes.append(epochTime)
                ActiveTrainingInfo.addEpochTime(epochTime)
                print(f'Epoch duration: {epochTime}')

            endTime = timer()
            trainingTimeStr = TimeUtils.getTimeDiffStr(startTime, endTime)
            ActiveTrainingInfo.setEndTime(TimeUtils.getCurrentTime())
            ActiveTrainingInfo.setStatus(TrainingStatus.FINISHED)
            print('Finished Training')
            print(f'Training duration: {trainingTimeStr}')

            self.classification.saveModel()

            self.test()

            ActiveTrainingInfo.saveResultData()
        except Exception as exception:
            ActiveTrainingInfo.setError(exception)
            ActiveTrainingInfo.saveResultData()
            raise exception

    def trainStep(self, dataLoader) -> None:
        size = len(dataLoader.dataset)
        runningLoss = 0.0
        for batch, data in enumerate(dataLoader, 0):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.network(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            runningLoss += loss.item()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(inputs)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def validationStep(self, dataLoader):
        size = len(dataLoader.dataset)
        numBatches = len(dataLoader)
        testLoss = 0.0
        correct = 0.0

        with torch.no_grad():
            for data in dataLoader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                labels = torch.flatten(labels)
                outputs = self.network(images)

                testLoss += self.criterion(outputs, labels).item()
                correct += (outputs.argmax(1) ==
                            labels).type(torch.float).sum().item()

        testLoss /= numBatches
        correct /= size

        self.lossData.append(testLoss)
        self.accuracyData.append(correct)

        ActiveTrainingInfo.addRunningLossData(testLoss)
        ActiveTrainingInfo.addRunningAccuracyData(correct)

        print(
            f'Test error: \n Accuracy: {(100 * correct):>0.1f} %, Avg loss: {testLoss:>8f} \n')
