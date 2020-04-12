import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd



def categorizeClasses(a, classes):
    return classes[a]

cat = np.vectorize(categorizeClasses)

# Create DataLoader
class TimeSeriesDataSet(Dataset):
    """
        TimeSeriesDataSet is a class that load a time series dataset
    """
    def __init__(self, data, label, id_poly=None):
        """
            data : numpy array
        """
        self.data = data
        self.label = label
        self.class_to_idx = {l:i for i, l in enumerate(set(label))}
        #print(self.class_to_idx)

        self.y_true = cat(self.label, self.class_to_idx) 
        #print(set(self.y_train))

        if id_poly is not None:
            self.id_poly = id_poly
        else:
            self.id_poly = None

        self.nb_samples = data.shape[0]
        self.len_timeseries = data.shape[1]
        self.nb_classes = len(set(label))
        
    def __len__(self):
        return self.nb_samples
    
    def __getitem__(self, idx):
        if self.id_poly is None:
            return self.data[idx], int(self.y_true[idx])
        else:
            return self.data[idx], int(self.y_true[idx]), self.id_poly[idx]

def computingMinMax(X, per=2):
	min_per = np.percentile(X, per, axis=(0,1))
	max_per = np.percentile(X, 100-per, axis=(0,1))
	return min_per, max_per

def normalizingData(X, min_per, max_per):
	return (X-min_per)/(max_per-min_per)

def readSITS_csv(path):
    """
        This function permet d'ouvrir un csv et retourner la matrice de la serie temporelle et le vecteur des labels
                                                             -----------------------------        ------------------
    """
    data = pd.read_csv(path, sep=",", header=None)

    y_data = data.iloc[:, 0]
    y_data = np.asarray(y_data.values, dtype='int64')

    poly_id = data.iloc[:, 1]
    poly_id = np.asarray(poly_id, dtype=np.float)

    X_data = data.iloc[:, 2:]
    X_data = np.asarray(X_data.values)
    X_data = X_data.reshape((X_data.shape[0], X_data.shape[1]//3, 3))

    # Normalisation des données -- Normalisation min/max 
    min_per, max_per = computingMinMax(X_data, per=2)
    X_data = normalizingData(X_data, min_per, max_per)

    return X_data, y_data, poly_id

def getDataLoader_fromCSV(path, **argv):

    batch_size = argv.setdefault("batch_size", 4)
    shuffle = argv.setdefault("shuffle", False)
    pin_memory = argv.setdefault("pin_memory", False)

    X_data, y_data, poly_id = readSITS_csv(path)

    dataSet = TimeSeriesDataSet(X_data, y_data, poly_id)

    dataLoader = DataLoader(dataSet, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return dataLoader


def getDataLoader(**argv):
    """
        inputs : data => table of data
                 lebl => labels of the data
    """

    batch_size = argv.setdefault("batch_size", 4)
    shuffle = argv.setdefault("shuffle", False)
    pin_memory = argv.setdefault("pin_memory", False)

    X_data = argv["data"] 
    y_data = argv["label"]

    dataSet = TimeSeriesDataSet(X_data, y_data)

    dataLoader = DataLoader(dataSet, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return dataLoader




def test_openDataset():
    from tslearn.datasets import UCR_UEA_datasets
    


    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("GunPoint")
    print("Train shape", X_train.shape)
    print("Test shape", X_test.shape)
    print(set(y_train))

    dl = getDataLoader(data=X_train, label=y_train)
    print("nb data", len(dl.dataset))

    plt.plot(range(X_train.shape[1]), X_train[0, :]/X_train.max())
    plt.plot(range(X_train.shape[1]), X_train[25, :]/X_train.max())

    plt.show()

if __name__ == "__main__":
    
    path = "C:/Users/mchelali/Documents/STIS/Interpoler/PlanerRep/csvData/"

    dl = getDataLoader_fromCSV(path+"1/test_dataset.csv")

    for d in dl:
        print(d[0].shape, d[1], d[2])
        break

    print("nb data", len(dl.dataset))
    print("data shape ", dl.dataset.data.shape)
    print("nb classes ", dl.dataset.nb_classes)
    print("nb classes ", dir(dl.dataset))

    plt.plot(range(dl.dataset.data.shape[1]), dl.dataset.data[0, :, 0],  c='red')
    plt.plot(range(dl.dataset.data.shape[1]), dl.dataset.data[0, :, 1],  c='red')
    plt.plot(range(dl.dataset.data.shape[1]), dl.dataset.data[0, :, 2],  c='red')

    plt.plot(range(dl.dataset.data.shape[1]), dl.dataset.data[25, :, 0], c='blue')
    plt.plot(range(dl.dataset.data.shape[1]), dl.dataset.data[25, :, 1], c='blue')
    plt.plot(range(dl.dataset.data.shape[1]), dl.dataset.data[25, :, 2], c='blue')

    plt.show()