import torch
import matplotlib.pyplot as plt 
import numpy as np 
from models import LSTM_Classifier
import dataloader as dl 
import tools

def crossvald():
    path = "C:/Users/mchelali/Documents/STIS/Interpoler/PlanerRep/csvData/"

    classes1 = ["Verger intensif", "Verger tradi"]


    times = 5
    nb_epochs = 200
    patience = 20
    batch_size = 64
    n_layers = 1
    for t in range(1, times+1):
        f = path + str(t) + "/"

        trainloader = dl.getDataLoader_fromCSV(f+"train_dataset.csv", batch_size=batch_size, shuffle=True, pin_memory=True)
        valdloader = dl.getDataLoader_fromCSV(f+"vald_dataset.csv", batch_size=batch_size, shuffle=True, pin_memory=True)
        testloader = dl.getDataLoader_fromCSV(f+"test_dataset.csv", batch_size=1, shuffle=False, pin_memory=True)

        seq_length = trainloader.dataset.data.shape[1]
        n_features = trainloader.dataset.data.shape[2]

        net = LSTM_Classifier(n_features=n_features, seq_length=seq_length, n_layers=n_layers, nb_classes=len(classes1))

        tools.trainLSTM(net=net, trainLoader=trainloader, valdLoader=valdloader, nb_epoch=nb_epochs, patience=patience, path=f, modelname="lstm_"+str(n_layers)+"layers.pth")

        bestEpoch = tools.plotLossCurves(path2model=f, figsize=None, netname="lstm_"+str(n_features)+"layers.pth")

        print("Meuilleur epoch est ", bestEpoch)


        tools.testLSTM(net=net, testLoader=testloader, classes=classes1, path=f)

if __name__ == "__main__":
    crossvald()