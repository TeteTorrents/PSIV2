# %%
# torch imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import WindowLevel

import os
import plotly.subplots
import plotly.graph_objects as go
import numpy as np
import json

import pandas as pd
#
from sklearn.model_selection import GroupKFold, train_test_split, KFold

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import plotly.subplots
import plotly.graph_objects as go

LR = 0.01
FUSION_METHOD = 'mean'
EPOCHS = 50
BATCH = 512
FOLDS = 5
SPLIT_LEVEL = 'seizure'

# %%
# data fusion unit
def data_fusion(method='concat'):
    if method == 'concat':
        return lambda data: data.reshape(data.size(1), -1)
    elif method == 'weighted_mean':
        return 1
    elif method == 'mean':
        return lambda data: torch.mean(data, dim=0)



# %%
# weighted_mean = nn.Conv1d(128,1, 1)
# tense = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).float()
# print(tense.shape)
# tense = tense.unsqueeze(0)
# print(tense.shape)
# print(weighted_mean(tense))




# %%
# define the model
class CNN(nn.Module):
    def __init__(self, data_fusion_method='concat'):
        super(CNN, self).__init__()
        # define the layers
        # self.conv1 = nn.Conv1d(128*21, 128*21//4, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(128*21//4, 128*21//16, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(128*21//16, 128*21//64, 3, stride=1, padding=1)
        if data_fusion_method == 'concat':
            input_channels = 128 * 21
        else:
            input_channels = 128
        self.data_fusion_method = data_fusion_method
        self.weighted_mean = nn.Conv1d(21, 1, 1)
        self.conv1 = nn.Conv1d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, stride=1, padding=1)
        self.pool = nn.ReLU()
        self.relu = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(64 * (input_channels // 8), 128)
        self.fc2 = nn.Linear(128, 2)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # data fusion
        func = data_fusion(self.data_fusion_method)
        
        if func == 1:
            # print(x.shape)
            x = x.permute(1,0,2)
            # print(x.shape)
            x = self.weighted_mean(x)
            x = x.squeeze(1)
            # print(x.shape)
        else:
            x = func(x)
        

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
    
    # TODO: Fer un refactoring d'aquesta funció per tal que s'entengui més.
    # Fer-ho fora de la classe model
    def train_(self, data_loader, epochs=10, lr=0.001):
        # define the loss function
        loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1/0.87, 1/0.13]).to('cuda'))
        # define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(params=self.parameters(), lr=0.001, momentum=0.9)
        # init the weights
        
        # train the model
        metrics = {'epoch': [], 'acc': [], 'precision': [], 'recall':[], 'f1': [], 'loss': []}
        for epoch in range(epochs):
            # print(len(data_loader))
            epoch_outputs = []
            epoch_labels = []
            # epoch_acc = []
            # epoch_recall = []
            epoch_loss = 0
            for i, data in enumerate(data_loader):
                # print(data[0])
                # print(data[0].shape)
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                # print("inputs.shape", inputs.shape)
                inputs = inputs.permute(1, 0, 2)
                # print("inputs.permute.shape", inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                # print(f"squeeze(0) {inputs.squeeze(0).shape}")
                outputs = self.forward(inputs)
                epoch_outputs.append(outputs.to('cpu').detach().numpy())
                epoch_labels.append(labels.to('cpu').detach().numpy())
                # print(labels.to('cpu').detach().numpy())
                # print(np.argmax(outputs.to('cpu').detach().numpy(), axis=1))
                # labels_batch = np.argmax(labels.to('cpu').detach().numpy(),axis=1)
                # outputs_batch = np.argmax(outputs.to('cpu').detach().numpy(), axis=1)
                # print(labels_batch)
                # print(outputs_batch)
                # epoch_recall.append(recall_score(labels_batch,outputs_batch))
                # outputs = outputs.to('cuda')
                
                # print("output train func:",outputs)
                # print(labels)
                # loss = loss_fn(outputs, torch.argmax(labels, dim=1))
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                epoch_loss += loss.item()
                if i % 600 == 0:
                    
                    continue
                    print("outputs",outputs)
                    print("labels",labels)
            # print(epoch_labels)
            # print(epoch_outputs)
            epoch_labels, epoch_outputs = np.argmax(np.concatenate(epoch_labels), axis=1), np.argmax(np.concatenate(epoch_outputs), axis=1)
            acc = accuracy_score(epoch_labels, epoch_outputs)
            precision = precision_score(epoch_labels, epoch_outputs, zero_division=0)
            recall = recall_score(epoch_labels, epoch_outputs)
            f1 = f1_score(epoch_labels, epoch_outputs)
            metrics['epoch'].append(epoch)
            metrics['acc'].append(acc)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['loss'].append(epoch_loss/len(data_loader))
            print(f'epoch: {epoch}, acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, loss: {epoch_loss/len(data_loader)}')
            # print("recall mean", np.array(epoch_recall).mean())
            # TEST
            self.test(data_loader)
        
        print('Finished Training')
        return metrics
    
    # TODO S'ha de fer un refactoring d'aquesta funció per tal que s'entengui més.
    # Fer-ho fora de la classe model
    def test(self, data_loader):
        self.eval()
        metrics = { 'acc': [], 'precision': [], 'recall':[], 'f1': []} 
        all_outputs = []
        all_labels = []
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            inputs = inputs.permute(1, 0, 2)
            outputs = self.predict(inputs)
            all_outputs.append(outputs.to('cpu').detach().numpy())
            all_labels.append(labels.to('cpu').detach().numpy())
        all_labels, all_outputs = np.argmax(np.concatenate(all_labels), axis=1), np.argmax(np.concatenate(all_outputs), axis=1)
        acc = accuracy_score(all_labels, all_outputs)
        precision = precision_score(all_labels, all_outputs, zero_division=0)
        recall = recall_score(all_labels, all_outputs)
        f1 = f1_score(all_labels, all_outputs)
        metrics['acc'].append(acc)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        print(f'acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}')
        return acc, precision, recall, f1

# %%
def train_model(model, data_loader, loss_fn, optimizer, epochs=10, lr=0.001):
    # train the model
    train_metrics = {'epoch': [], 'acc': [], 'precision': [], 'recall':[], 'f1': [], 'loss': [], 'confmat': []}
    test_metrics = {'epoch': [], 'acc': [], 'precision': [], 'recall':[], 'f1': [], 'loss': [], 'confmat': []}
    train_confusion_matrixes = []
    test_confusion_matrixes = [] 
    for epoch in range(epochs):
        
        train_outputs = []
        train_labels = []
        test_outputs = []
        test_labels = []
        train_loss = 0.0
        test_loss = 0.0
        model.train()
        for i, data in enumerate(data_loader):
            
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            inputs = inputs.permute(1, 0, 2)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            train_outputs.append(outputs.to('cpu').detach().numpy())
            train_labels.append(labels.to('cpu').detach().numpy())

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            train_loss += loss.item()
            if i % 600 == 0:
                
                continue
        train_labels, train_outputs = np.argmax(np.concatenate(train_labels), axis=1), np.argmax(np.concatenate(train_outputs), axis=1)
        
        
        acc = accuracy_score(train_labels, train_outputs)
        precision = precision_score(train_labels, train_outputs, zero_division=0)
        recall = recall_score(train_labels, train_outputs)
        f1 = f1_score(train_labels, train_outputs)
        loss = train_loss/len(data_loader)
        train_confmat = confusion_matrix(train_labels, train_outputs)
        train_confusion_matrixes.append(train_confmat)

        train_metrics['epoch'].append(epoch)
        train_metrics['acc'].append(acc)
        train_metrics['precision'].append(precision)
        train_metrics['recall'].append(recall)
        train_metrics['f1'].append(f1)
        train_metrics['loss'].append(loss)
        train_metrics['confmat'].append(train_confmat.tolist())
        
        

        print(f'TRAIN --- epoch: {epoch}, acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, loss: {loss}')
        
        # EVAL
        model.eval()
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            inputs = inputs.permute(1, 0, 2)
            outputs = model.forward(inputs)
            test_outputs.append(outputs.to('cpu').detach().numpy())
            test_labels.append(labels.to('cpu').detach().numpy())
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            
        test_labels, test_outputs = np.argmax(np.concatenate(test_labels), axis=1), np.argmax(np.concatenate(test_outputs), axis=1)
        
        acc = accuracy_score(test_labels, test_outputs)
        precision = precision_score(test_labels, test_outputs, zero_division=0)
        recall = recall_score(test_labels, test_outputs)
        f1 = f1_score(test_labels, test_outputs)
        loss = test_loss/len(data_loader)
        test_confmat = confusion_matrix(test_labels, test_outputs)
        test_confusion_matrixes.append(test_confmat)

        test_metrics['epoch'].append(epoch)
        test_metrics['acc'].append(acc)
        test_metrics['precision'].append(precision)
        test_metrics['recall'].append(recall)
        test_metrics['f1'].append(f1)
        test_metrics['loss'].append(loss)
        test_metrics['confmat'].append(test_confmat.tolist())

        

        
        print(f'TEST --- epoch: {epoch}, acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, loss: {loss}')
        print('train confusion matrix')
        print(train_confmat)
        print('test confusion matrix')
        print(test_confmat)
        print('--------------------------------------')
        
    print('Finished Training')
    metrics = {'train': train_metrics, 'test': test_metrics}
    return metrics


def new_windu_from_index(windu, indexes):
    new_windu = WindowLevel()
    new_windu.windows = np.array(windu.windows[indexes])
    new_windu.labels = np.array(windu.labels[indexes])
    new_windu.groups = np.array(windu.groups)[indexes]
    return new_windu


def plot_fold_metrics(metriques, metric_type='loss'):
    

    if metric_type == 'confmat':
        fig = plotly.subplots.make_subplots(rows=1, cols=2, subplot_titles=[f' {i} cumulative confmat' for i in ['train','test']], horizontal_spacing=0.15)

        cum_confmat_train = np.zeros_like(metriques[0]['train']['confmat'][0])
        cum_confmat_test = np.zeros_like(metriques[0]['train']['confmat'][0])
        # Add training confusion matrix heatmap
        
        for i, fold in enumerate(metriques):
            train_metrics, test_metrics = fold['train'], fold['test']
            print(train_metrics)
            train_confmat = np.array(train_metrics['confmat'])[-1]
            test_confmat = np.array(test_metrics['confmat'])[-1]
            cum_confmat_train += train_confmat
            cum_confmat_test += test_confmat
        fig.add_trace(go.Heatmap(z=cum_confmat_train, colorscale='Blues', colorbar_x=0.45),
                        row=1, col=1)

        # Add testing confusion matrix heatmap
        fig.add_trace(go.Heatmap(z=cum_confmat_test, colorscale='Oranges'),
                        row=1, col=2)
        # Add labels
        fig.update_yaxes(title_text=f'{metric_type.capitalize()}', row=1, col=1)
        fig.update_yaxes(title_text=f'{metric_type.capitalize()}', row=1, col=2)

        # Add legend for the entire figure
        # fig.update_layout(legend=dict(traceorder='normal', orientation='h', y=-0.2))

        fig.update_layout(
            width=800,  # Adjust the width as needed
            height=440,  # Adjust the height as needed
        )

        return fig
        
    num_folds = len(metriques)

    # Create subplots with 2 rows and 'num_folds' columns
    fig = plotly.subplots.make_subplots(rows=num_folds, cols=1, subplot_titles=[f'Fold {i}' for i in range(1, num_folds + 1)])

    # Define a color map for the traces
    color_map = {'Train': 'blue', 'Test': 'orange'}
    for i, fold in enumerate(metriques, 1):
        train_metrics, test_metrics = fold['train'], fold['test']
        
        
       
        # Select the metric type (e.g., 'loss', 'accuracy', etc.)
        train_metric_values = train_metrics.get(metric_type, [])
        test_metric_values = test_metrics.get(metric_type, [])

        # Add training plot with specified color and legend group
        fig.add_trace(go.Scatter(x=train_metrics['epoch'], y=train_metric_values, mode='lines', name='Train',
                                line=dict(color=color_map['Train']), legendgroup=f'fold{i}'),
                    row=i, col=1)

        # Add testing plot with specified color and legend group
        fig.add_trace(go.Scatter(x=test_metrics['epoch'], y=test_metric_values, mode='lines', name='Test',
                                line=dict(color=color_map['Test']), legendgroup=f'fold{i}'),
                    row=i, col=1)

        # Set the x-axis ticks to integers in the range 0 to 19
        fig.update_xaxes(tickmode='array', tickvals=list(range(EPOCHS)), row=i, col=1)

        # Add labels
        fig.update_yaxes(title_text=f'{metric_type.capitalize()}', row=i, col=1)

        # Add legend for the entire figure
        fig.update_layout(legend=dict(traceorder='normal', orientation='h', y=-0.2))

        # Set the aspect ratio to make the subplots square
        fig.update_layout(
            width=800,  # Adjust the width as needed
            height=400 * num_folds,  # Adjust the height as needed
        )

    # Show the plot
    return fig

if __name__ == '__main__':
    print('starting...')
    # %%
    trans = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Lambda(lambda x: np.reshape(x, (1, -1)))
        
    ])

    # %% [markdown]
    # # NIVELL WINDOW
    

    # %%
    print('loading data...')
    windu = WindowLevel('../data/annotated_windows/', transforms=trans, split_level = SPLIT_LEVEL)
    data_loader = DataLoader(windu, batch_size=512, shuffle=True)

    # %%
    X = windu.windows
    y = windu.labels

    # %%
    
    # per a 'mean' dona molt bons resultats amb LR de 0.001 amb crossentropyloss, Adam d'optimizer, batch de 512 i 20 epochs
    # amb LR de 0.001 el 'concat' dona resultats de merda

    grups = windu.groups
    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    models = []
    metriques = []
    print('starting training...')
    for fold, (train_index, val_index) in enumerate(skf.split(X)):
        print(f'############### Fold {fold} ###############')
        model = CNN(data_fusion_method=FUSION_METHOD).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1/0.87, 1/0.13]).to('cuda'))
        # print(model)
        # for param in model.parameters():
        #     if param.dim() > 1:
        #         # kaiming init
        #         nn.init.kaiming_normal_(param)
            # nn.init.xavier_uniform_(model.weighted_mean.weight)

        train_windu = new_windu_from_index(windu, train_index)
        val_windu = new_windu_from_index(windu, val_index)
        train_data_loader = DataLoader(train_windu, batch_size=BATCH, shuffle=True)
        val_data_loader = DataLoader(val_windu, batch_size=BATCH, shuffle=True)
        metrics = train_model(model, train_data_loader, loss_fn, optimizer, epochs=EPOCHS, lr=LR)
        models.append(model)
        metriques.append(metrics)
    
    print('training finished!!!!!!!!!!!!!!')


    # %%



    # create fold for subplots with random name

    #mkdir a random name
    # get the folders highest number
    # create a folder with that number + 1
    # create the plots inside that folder
    os.listdir('plots/')
    if len(os.listdir('plots/')) == 0:
        folder_name = f'{SPLIT_LEVEL}_{FUSION_METHOD}_execucio_1'
        os.mkdir(f'plots/{folder_name}')
        os.mkdir(f'models/{folder_name}')
        
    else:
        folder_name = f'{SPLIT_LEVEL}_{FUSION_METHOD}_execucio_{len(os.listdir("plots/")) + 1}'
        os.mkdir(f'plots/{folder_name}')
        os.mkdir(f'models/{folder_name}')

    print('folder name',folder_name)

    # save the models
    print('saving models...')
    for i, m in enumerate(models):
        torch.save(m.state_dict(), f'models/{folder_name}/model_fold_{i}.pt')

    # save the dict metrics as a json
    print('saving metrics...')
    with open(f'models/{folder_name}/metrics.json', 'w') as fp:
        json.dump(metriques, fp)

    print('saving plots...')
    fig_loss = plot_fold_metrics(metriques, metric_type='loss')
    # Save the plot as png
    fig_loss.write_image(f'plots/{folder_name}/loss.png')

    fig_acc = plot_fold_metrics(metriques, metric_type='acc')
    # Save the plot as png
    fig_acc.write_image(f'plots/{folder_name}/acc.png')

    fig_precision = plot_fold_metrics(metriques, metric_type='precision')
    # Save the plot as png
    fig_precision.write_image(f'plots/{folder_name}/precision.png')

    fig_recall = plot_fold_metrics(metriques, metric_type='recall')
    # Save the plot as png
    fig_recall.write_image(f'plots/{folder_name}/recall.png')

    fig_f1 = plot_fold_metrics(metriques, metric_type='f1')
    # Save the plot as png
    fig_f1.write_image(f'plots/{folder_name}/f1.png')

    fig_confmat = plot_fold_metrics(metriques, metric_type='confmat')
    # Save the plot as png
    fig_confmat.write_image(f'plots/{folder_name}/confmat.png')

    print("THE END")