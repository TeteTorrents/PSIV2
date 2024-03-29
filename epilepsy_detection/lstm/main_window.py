import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from model_lstm import EpilepsyLSTM
from model_transformer import EpilepsyTransformer
from dataset import WindowLevel
from torch.utils.data import Subset
from tqdm import tqdm

### DEFINE VARIABLES
DEVICE = 'cuda:0'       # options: 'cpu', 'cuda:0', 'cuda:1'
N_CLASSES = 2        # number of classes. This case 2={seizure ,non-seizure}
print("TRANSFORMERRRR")
# Default hyper parameters
def get_default_hyperparameters():
   
    # initialize dictionaries
    inputmodule_params={}
    net_params={}
    outmodule_params={}
    
    # network input parameters
    inputmodule_params['n_nodes'] = 21
    
    # LSTM unit  parameters
    net_params['Lstacks'] = 1  # stacked layers (num_layers)
    net_params['dropout'] = 0.0
    net_params['hidden_size']= 256  #h
   
    # network output parameters
    outmodule_params['n_classes'] = 2
    outmodule_params['hd'] = 128
    
    return inputmodule_params, net_params, outmodule_params

# Create the LSTM model
inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
lstm_model = EpilepsyTransformer(inputmodule_params, net_params, outmodule_params)
lstm_model.init_weights()
lstm_model.to(DEVICE)


# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Create the WindowLevel dataset
dataset = WindowLevel(data_dir=r'/fhome/mapsiv02/Desktop/projecte_epil/data/annotated_windows', split_level='seizure')
print(f'LENGTH DATASET: {len(dataset)}')

# Prepare data for stratified k-fold
X = dataset.windows
y = dataset.labels
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics for each fold
all_precisions = []
all_recalls = []
all_f1_scores = []
all_accuracies = []
all_conf_matrices = []

# Stratified k-fold cross-validation
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    # Split the data
    train_data, val_data = X[train_index], X[val_index]
    train_labels, val_labels = y[train_index], y[val_index]

    # Convert data to PyTorch DataLoader
    train_subset = Subset(dataset, train_index)
    val_subset = Subset(dataset, val_index)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)

    # Training loop
    num_epochs = 25
    for epoch in range(num_epochs):
        train_bar = tqdm(train_loader, desc=f'Fold {fold + 1}, Epoch {epoch + 1}, Training')
        total_loss = 0.0
        for inputs, labels in train_bar:
            # Forward pass
            outputs = lstm_model(inputs.to(DEVICE))
            #out, (hn, cn) = lstm_model.lstm(inputs)

            # Compute loss
            loss = criterion(outputs.squeeze().cpu(), labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'Loss': loss.item()})

        average_loss = total_loss / len(train_loader.dataset)
        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Average Training Loss: {average_loss:.4f}')
        val_bar = tqdm(val_loader, desc=f'Fold {fold + 1}, Epoch {epoch + 1}, Validation')

        # Evaluate on the validation set
        all_preds = []
        all_targets = []
        total_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_bar:
                outputs = lstm_model(inputs.to(DEVICE))
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                all_preds.extend(predictions.cpu().detach().numpy())
                all_targets.extend(torch.argmax(labels, dim=1).numpy())
                val_loss = criterion(outputs.cpu(), labels)
                total_val_loss += val_loss.item() * inputs.size(0)

        # Calculate evaluation metrics
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        accuracy = accuracy_score(all_targets, all_preds)
        conf_matrix = confusion_matrix(all_targets, all_preds)

        # Store metrics for this fold
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        all_accuracies.append(accuracy)
        all_conf_matrices.append(conf_matrix)

        # Print or store the metrics for this fold as needed
        average_val_loss = total_val_loss / len(val_loader.dataset)
        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Average Validation Loss: {average_val_loss:.4f}')
        print(f'Fold {fold + 1}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}, Accuracy={accuracy:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)
        print("...")

# Print or store the average metrics across all folds
avg_precision = sum(all_precisions) / len(all_precisions)
avg_recall = sum(all_recalls) / len(all_recalls)
avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
avg_accuracy = sum(all_accuracies) / len(all_accuracies)

print('\nAverage Metrics Across Folds:')
print(f'Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1-Score={avg_f1:.4f}, Accuracy={avg_accuracy:.4f}')

torch.save(lstm_model.state_dict(), 'transformer_model.pth')
print("Trained model saved.")