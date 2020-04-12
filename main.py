import torch
import matplotlib.pyplot as plt 
import numpy as np 
from models import LSTM_Classifier
import dataloader as dl 
import tools
import os 

def crossvald():
    path = "C:/Users/mchelali/Documents/STIS/csv_data/"

    times = 5
    nb_epochs = 200
    patience = 20
    batch_size = 256
    n_layers = 2
    for t in range(1, times+1):
        os.makedirs(path + str(t) + "/lstm_"+str(n_layers)+ "/", exist_ok=True)
        savepath = path + str(t) + "/lstm_"+str(n_layers)+ "/"
        f = path + str(t) + "/"

        trainloader = dl.getDataLoader_fromCSV(f+"train_dataset.csv", batch_size=batch_size, shuffle=True, pin_memory=True)
        valdloader = dl.getDataLoader_fromCSV(f+"vald_dataset.csv", batch_size=batch_size, shuffle=True, pin_memory=True)
        testloader = dl.getDataLoader_fromCSV(f+"test_dataset.csv", batch_size=1, shuffle=False, pin_memory=True)

        seq_length = trainloader.dataset.len_timeseries
        n_features = trainloader.dataset.data.shape[2]

        print("seq len ", seq_length, testloader.dataset.len_timeseries)
        print("nb featires ", n_features)
        print("nb classes ", testloader.dataset.nb_classes)

        classes1 = list(trainloader.dataset.class_to_idx.keys())
        print(classes1)
        #exit(-1)

        net = LSTM_Classifier(n_features=n_features, seq_length=seq_length, n_layers=n_layers, nb_classes=len(classes1))

        if not os.path.exists(savepath + "lstm_"+str(n_layers)+"layers.pth"):
            tools.trainLSTM(net=net, trainLoader=trainloader, valdLoader=valdloader, nb_epoch=nb_epochs, patience=patience, path=savepath, modelname="lstm_"+str(n_layers)+"layers.pth")

        bestEpoch = tools.plotLossCurves(path2model=savepath, figsize=None, netname="lstm_"+str(n_layers)+"layers.pth")

        print("Meuilleur epoch est ", bestEpoch)

        checkpoint = torch.load(savepath+"lstm_"+str(n_layers)+"layers.pth")
        net.load_state_dict(checkpoint["bestModel"])
        net.eval()
        tools.testLSTM(net=net, testLoader=testloader, classes=classes1, path=savepath)

if __name__ == "__main__":
    crossvald()