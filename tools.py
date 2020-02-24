import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os 


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def computingConfMatrix(referenced, p_test, n_classes):
    """
        Computing a n_classes by n_classes confusion matrix
        INPUT:
            - referenced: reference data labels,one hot encoding of the test labels
            - p_test: predicted 'probabilities' from the model for the test instances
            - n_classes: number of classes (numbered from 0 to 1)
        OUTPUT:
            - C: computed confusion matrix
	"""
    predicted = p_test.argmax(axis=1)
    referenced = referenced.argmax(axis=1)
    C = np.zeros((n_classes, n_classes))
    for act, pred in zip(referenced, predicted):
        C[act][pred] += 1
    return C

def computingConfMatrixperPolygon(y_test, p_test, polygon_ids_test, n_classes):
    """
        Computing a n_classes by n_classes confusion matrix
        INPUT:
            - y_test_one_one: one hot encoding of the test labels
            - p_test: predicted 'probabilities' from the model for the test instances
            - n_classes: number of classes (numbered from 0 to 1)
        OUTPUT:
            - C_poly_perpoly: computed confusion matrix at polygon level with polygon count
            - C_poly_perpix: computed confusion matrix at polygon level with pixel count
            - OA_poly_poly: OA at polygon level with polygon count
            - OA_poly_pix: OA at polygon level with pixel count
    """

    nbTestInstances = y_test.shape[0]
    unique_pol_test = np.unique(polygon_ids_test)
    n_polygons_test = len(unique_pol_test)
    C_poly_perpoly = np.zeros((n_classes, n_classes))
    C_poly_perpix = np.zeros((n_classes, n_classes))

    probas_per_polygon = {x: np.zeros(n_classes, dtype=float) for x in unique_pol_test}
    n_pixels_per_polygon = {x: 0 for x in unique_pol_test}
    for i in range(nbTestInstances):
        poly = polygon_ids_test[i]
        pred = p_test[i]
        probas_per_polygon[poly] = probas_per_polygon.get(poly) + pred
        n_pixels_per_polygon[poly] = n_pixels_per_polygon[poly] + 1

    for poly, probas in probas_per_polygon.items():
        probas_per_polygon[poly] = probas / n_pixels_per_polygon.get(poly)
        pred_class_id = np.argmax(probas_per_polygon[poly])

        id_line_with_right_poly = polygon_ids_test.tolist().index(poly)
        correct_class_index = np.argmax(y_test[id_line_with_right_poly])
        C_poly_perpoly[correct_class_index, pred_class_id] = C_poly_perpoly[correct_class_index, pred_class_id] + 1
        C_poly_perpix[correct_class_index, pred_class_id] = C_poly_perpix[correct_class_index, pred_class_id] + \
                                                            n_pixels_per_polygon[poly]

    OA_poly_poly = round(float(np.trace(C_poly_perpoly)) / n_polygons_test, 4)
    OA_poly_pix = round(float(np.trace(C_poly_perpix)) / nbTestInstances, 4)

    return C_poly_perpoly, C_poly_perpix, OA_poly_poly, OA_poly_pix

def save_confusion_matrix(C, class_name, conf_file):
    """
        Create a confusion matrix with IndexName, Precision, Recall, F-Score, OA and Kappa
        Charlotte's style
        INPUT:
            - C: confusion_matrix compute by sklearn.metrics.confusion_matrix
            - class_name: corresponding name class
        OUTPUT:
            - conf_mat: Charlotte's confusion matrix
    """

    nclass, _ = C.shape

    # -- Compute the different statistics
    recall = np.zeros(nclass)
    precision = np.zeros(nclass)
    fscore = np.zeros(nclass)
    diag_sum = 0
    hdiag_sum = 0
    for add in range(nclass):
        hdiag_sum = hdiag_sum + np.sum(C[add, :]) * np.sum(C[:, add])
        if C[add, add] == 0:
            recall[add] = 0
            precision[add] = 0
            fscore[add] = 0
        else:
            recall[add] = C[add, add] / np.sum(C[add, :])
            recall[add] = "%.6f" % recall[add]
            precision[add] = C[add, add] / np.sum(C[:, add])
            precision[add] = "%.6f" % precision[add]
            fscore[add] = (2 * precision[add] * recall[add]) / (precision[add] + recall[add])
            fscore[add] = "%.6f" % fscore[add]
    nbSamples = np.sum(C)
    OA = np.trace(C) / nbSamples
    ph = hdiag_sum / (nbSamples * nbSamples)
    kappa = (OA - ph) / (1.0 - ph)


    #################### PANDA DATAFRAME #####################

    writer = pd.ExcelWriter(conf_file)

    line = [' ']
    for name in class_name:
        line.append(name)
    line.append('Recall')
    line = pd.DataFrame(np.array(line).reshape((1,-1)))
    line.to_excel(writer, startrow=0, header=False, index=False)

    row_ = 1
    for j in range(nclass):
        line = [class_name[j]]
        for i in range(nclass):
            line.append(str(C[j, i]))
        line.append(str(recall[j]))  # + '\n'
        line = pd.DataFrame(np.array(line).reshape((1, -1)))
        line.to_excel(writer, startrow=row_, header=False, index=False)
        row_+=1

    line = ["Precision"]
    for add in range(nclass):
        line.append(str(precision[add]))
    line.append(str(OA))
    line.append(str(kappa))  # + '\n'
    line = pd.DataFrame(np.array(line).reshape((1, -1)))
    line.to_excel(writer, startrow=row_, header=False, index=False)
    row_ += 1

    line = ["F-Score"]
    for add in range(nclass):
        line.append(str(fscore[add]))
    line = pd.DataFrame(np.array(line).reshape((1, -1)))
    line.to_excel(writer, startrow=row_, header=False, index=False)

    writer.save()


def trainLSTM(net, trainLoader, valdLoader, **argv):

    nb_epoch = argv.setdefault("nb_epoch", 100)
    patience = argv.setdefault("patience", 0)
    path = argv.setdefault("path", "")
    modelname = argv.setdefault("modelname", "modelNN.pth")
    
    if torch.cuda.is_available():
        print("net to cuda")
        net.cuda()
    else:
        print("cuda is not available")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    ############################################################################
    #                         Initialisation des parametres                    #
    ############################################################################
    epoch = 0
    best_epoch = 0
    train_loss = []
    vald_loss = []
    epoch_list = []
    bestLoss = np.inf


    ############################################################################
    #                   Chargement du model si load est True                   #
    ############################################################################
    '''
        'epoch': epoch,
        'bestModel'
        'bestModel_optimizer'
        'lastModel'
        'lastModel_optimizer'
        'loss'
        'bestEpoch'
        'epochList'
        'listLoss'
        'lossValidation'
    '''
    if os.path.exists(path + "/" + modelname):
        checkpoint = torch.load(path + modelname)

        net.load_state_dict(checkpoint['lastModel'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        epoch_list = checkpoint['epochList']
        train_loss = checkpoint['listLoss']
        vald_loss = checkpoint['lossValidation']
        best_epoch = checkpoint["bestEpoch"]
        bestLoss = vald_loss[best_epoch]

        optimizer.load_state_dict(checkpoint['lastModel_optimizer'])
        print("Loading last model... ")

    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    while epoch < nb_epoch:

        loss = 0.
        nb_data = 0.
        net.train()
        for data in trainLoader:
            series, label, id_poly = data

            optimizer.zero_grad()

            net.init_hidden(series.shape[0], device)
            output = net(series.float().to(device))

            loss_ = criterion(output, label.to(device))
            loss += loss_.data.cpu().detach().numpy()

            loss_.backward()
            optimizer.step()

            nb_data += 1.
        
        loss = loss / nb_data
        train_loss.append(loss)


        loss_val = 0.
        nb_data = 0.
        net.eval()
        for data in valdLoader:
            series, label, id_poly = data

            net.init_hidden(series.shape[0], device)
            output = net(series.float().to(device))

            loss_ = criterion(output, label.to(device))
            loss_val += loss_.data.cpu().detach().numpy()   

            nb_data += 1.   
        loss_val = loss_val / nb_data
        vald_loss.append(loss_val)

        early_stopping(loss_val, net)

        epoch_list.append(epoch)

        if bestLoss > loss_val:
            best_epoch = epoch
            bestLoss = loss_val

            #####################################################
            #     Enregistrer le model pour cette epoch         #
            #####################################################

            torch.save({
                'epoch': epoch,
                'bestModel': net.state_dict(),
                'bestModel_optimizer': optimizer.state_dict(),
                'lastModel': net.state_dict(),
                'lastModel_optimizer': optimizer.state_dict(),
                'loss': bestLoss,
                'bestEpoch': best_epoch,
                'epochList': epoch_list,
                'listLoss': train_loss,
                'lossValidation': vald_loss,
                'earlyStop': early_stopping.early_stop
                },
                path + modelname)
        else:
            checkpoint = torch.load(path + modelname)

            torch.save({
                'epoch': epoch,
                'bestModel': net.state_dict(),
                'bestModel_optimizer': optimizer.state_dict(),
                'lastModel': net.state_dict(),
                'lastModel_optimizer': optimizer.state_dict(),
                'loss': bestLoss,
                'bestEpoch': best_epoch,
                'epochList': epoch_list,
                'listLoss': train_loss,
                'lossValidation': vald_loss,
                'earlyStop': early_stopping.early_stop
                },
                path + modelname)

        epoch += 1

        if early_stopping.early_stop:
            print("Early stopping")
            break
        print("Epoch ", epoch, "Train loss ", loss, "; Validation loss ", loss_val)
    print('Finished Training')


def testLSTM(net, testLoader, **argv):
    classes = argv.setdefault("classes", [str(i) for i in testLoader.dataset.nb_classes])
    path = argv.setdefault("path", "")

    print("nb exemple de test : ", len(testLoader.dataset))
    p_pred = np.zeros((len(testLoader.dataset), testLoader.dataset.nb_classes), dtype=float)  # probailité des classes pour chaque image
    p_true = np.zeros((len(testLoader.dataset), testLoader.dataset.label), dtype=int) # le label de la series ==> one hot encoder
    polyInfos = np.zeros(len(testLoader.dataset), dtype=float)
    


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("net to cuda")
        net.cuda()
    else:
        print("doesnt set net to cuda")

    net.eval()
    idx = 0
    for data in testLoader:
        series, label, id_poly = data
        
        
        polyInfos[idx] = id_poly
        p_true[idx, label] = 1 # One shot encoding
        net.init_hidden(series.shape[0], device)
        outputs = net(series.float().to(device))

        predicted = torch.softmax(outputs.data, 1).cpu().numpy()
        p_pred[idx, :] = predicted[0]

        idx+=1

    matConf = computingConfMatrix(p_true, p_pred, testLoader.dataset.nb_classes)
    C_poly_perpoly, C_poly_perpix, OA_poly_poly, OA_poly_pix = computingConfMatrixperPolygon(p_true, p_pred, polyInfos[:, 0], testLoader.dataset.nb_classes)

    save_confusion_matrix(matConf, classes, path + "rapport_classif_pixel.xlsx")
    save_confusion_matrix(C_poly_perpoly, classes, path + "rapport_classif_poly.xlsx")

def plotLossCurves(path2model, figsize=None, **param):
    """
        Cette fct permet de plotter les courbes du loss de train/validation est l'enregister dans le meme chemin que les models
    :param path2model: chemin ou les models de chaque epoch sont enregistrer
    :param figsize: la taille de la figure en sortie

    """

    netname = param.setdefault("netname", "modelNN.pth")

    '''
              'epoch': epoch,
              'bestModel'
              'bestModel_optimizer'
              'lastModel'
              'lastModel_optimizer'
              'loss'
              'bestEpoch'
              'epochList'
              'listLoss'
              'lossValidation'
    
    '''
    checkpoint = torch.load(path2model + netname )

    epoch = checkpoint['epoch']
    epoch_list = checkpoint['epochList']
    loss = checkpoint['loss']
    bestEpoch = checkpoint['bestEpoch']
    loss_list = checkpoint['listLoss']
    lossVald_list = checkpoint['lossValidation']

    print("Loading ... ")
    #print("lastt epoch ", epoch)
    #print(len(epoch_list))
    #print(len(loss_list))
    #print(len(lossVald_list))

    plt.figure(figsize=figsize)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.2)

    # plt.xticks(epoch_list, rotation=45)

    plt.plot(epoch_list, loss_list, label="loss train")
    plt.plot(epoch_list, lossVald_list, label="loss validation")
    plt.axvline(bestEpoch, c='red', linestyle='--', label='Early Stopping Checkpoint')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 1)

    plt.legend()
    # plt.show()
    plt.savefig(path2model + netname + ".png")

    return bestEpoch