import os
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix



#Normalize the data
def normalize_data(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler


#Load the data
def load_data():
    data = pd.read_csv(r'dados\dataset.csv')
    data = data.values
    X = data[:, 2:]
    y = data[:, 0]
    w = data[:, 1]
    return X, y, w


#Split the data
def split_data(X, y, w):
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, w_train, w_test


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class LogisticRegressionPersonalized(nn.Module):

    def __init__(self, criterion, optimizer, epochs, lr, scheduler=None):
        super(LogisticRegressionPersonalized, self).__init__()

        X, y, w = load_data()
        w = w + 1

        X, self.scaler = normalize_data(X)
        self.X_train, self.X_test, self.y_train,self.y_test, self.w_train, self.w_test = split_data(X, y, w)

        self.X_test = torch.from_numpy(self.X_test).float()
        self.y_test = torch.from_numpy(self.y_test).float()
        self.w_test = torch.from_numpy(self.w_test).float()

        input_dim = self.X_train.shape[1]
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.criterion = criterion
        self.epochs = epochs
        self.lr = lr

        self.threshold = 0.5

        if scheduler:
            self.scheduler = scheduler(self.optimizer, mode='min', factor=0.1, patience=10)

        self.apply(init_weights)

        
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
    

    def fit(self):
        X = torch.from_numpy(self.X_train).float()
        y = torch.from_numpy(self.y_train).float()
        w = torch.from_numpy(self.w_train).float()

        last_loss = 0
        max_epochs_no_improvement = 100
        epochs_no_improvement = 0

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.forward(X).squeeze()
            loss_unweighted = self.criterion(outputs, y)

            loss = (loss_unweighted * w).mean()
            loss.backward()
            self.optimizer.step()

            if abs(loss.item() - last_loss) < 1e-5:
                epochs_no_improvement += 1
            else:
                epochs_no_improvement = 0
                last_loss = loss.item()

            if epochs_no_improvement == max_epochs_no_improvement:
                print('Early stopping at epoch', epoch, 'Loss:', loss.item())
                break

            if epoch % 1000 == 0:
                print('Epoch:', epoch, 'Loss:', loss.item())

            if hasattr(self, 'scheduler'):
                self.scheduler.step(loss)


    def predict(self, X):
        X = torch.Tensor(X).float().reshape(1, -1)
        X = self.scaler.transform(X)
        X = torch.from_numpy(X).float()
        
        with torch.no_grad():
            y_pred_proba = torch.sigmoid(self.forward(X))
            y_pred_class = (y_pred_proba > self.threshold).long()
        return y_pred_class.numpy().reshape(-1), y_pred_proba.numpy().reshape(-1)


    def metrics(self, threshold=0.53):
        self.threshold = threshold
        # Previsões
        with torch.no_grad():
            y_pred_proba = torch.sigmoid(self.forward(self.X_test))  # Probabilidades
            y_pred_class = (y_pred_proba > threshold).long()  # Classes

        # Converter tensores para numpy
        y_test_np = self.y_test.numpy()
        y_pred_class_np = y_pred_class.numpy()
        weights_np = self.w_test.numpy()

        # Calcular métricas ponderadas
        accuracy = accuracy_score(y_test_np, y_pred_class_np, sample_weight=weights_np)
        precision = precision_score(y_test_np, y_pred_class_np, sample_weight=weights_np, zero_division=0)
        recall = recall_score(y_test_np, y_pred_class_np, sample_weight=weights_np, zero_division=0)
        conf_matrix = confusion_matrix(y_test_np, y_pred_class_np)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix
        }


class SklearnLogisticRegression():

    def __init__(self, *args, **kwargs):
        self.model = LogisticRegression(*args, **kwargs)
        X, y, w = load_data()
        w = w + 1
        X, self.scaler = normalize_data(X)
        self.X_train, self.X_test, self.y_train, self.y_test, self.w_train, self.w_test = split_data(X, y, w)

        self.threshold = 0.5


    def fit(self):
        self.model.fit(self.X_train, self.y_train, sample_weight=self.w_train)

    
    def predict(self, X):
        X = torch.Tensor(X).float().reshape(1, -1)
        X = self.scaler.transform(X)
        X = torch.from_numpy(X).float()
        
        with torch.no_grad():
            lables = self.model.predict(X)
            y_pred_class = (lables > self.threshold).astype(int)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
        return y_pred_class, y_pred_proba


    def metrics(self, threshold=0.53):
        self.threshold = threshold
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred_class = (y_pred_proba > threshold).astype(int)

        accuracy = accuracy_score(self.y_test, y_pred_class, sample_weight=self.w_test)
        precision = precision_score(self.y_test, y_pred_class, sample_weight=self.w_test, zero_division=0)
        recall = recall_score(self.y_test, y_pred_class, sample_weight=self.w_test, zero_division=0)
        conf_matrix = confusion_matrix(self.y_test, y_pred_class)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix
        }
 
    
model_exists = os.path.exists('model.pth')
if model_exists:
    try:
        model_personalized = torch.load('model.pth', weights_only=False)
    except Exception as e:
        print('Erro ao carregar o modelo:', e)
        model_exists = False


if not model_exists:  
    model_personalized = LogisticRegressionPersonalized( criterion=nn.BCELoss(), 
                                optimizer=torch.optim.Adam,
                                epochs=100000, lr=0.01,
                                scheduler=ReduceLROnPlateau)
    model_personalized.fit()
    torch.save(model_personalized, 'model.pth')

model_sklearn = LogisticRegression()
model_sklearn = SklearnLogisticRegression(solver='lbfgs')
model_sklearn.fit()

def get_model(personalized = True):
    if personalized: return model_personalized
    return model_sklearn


