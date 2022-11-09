# Code cell intended for imports and global settings.

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from SupportClasses import ModelSupport as ms, EnvironmentSetup as env

# Global values
NUMBER_OF_NODES = 5
NUMBER_OF_EPOCHS = 25
best_valid_loss = float('inf')

# Model specific values
criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":
    # This code cell is to be used for importing data and setting up the model.
    train_loader, validation_loader, test_loader = env.download_mnist()
    # Create and load the model
    model = ms.ConvNet(len(train_loader.dataset.dataset.classes))
    models = [model]

    # Send the models to the CUDA device if it exists.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models:
        model.to(device=device)

    optimizer = optim.Adam(model.parameters())
    for epoch in range(NUMBER_OF_EPOCHS):

        train_loss, train_acc = ms.train(model, train_loader, optimizer, criterion, device=device)
        valid_loss, valid_acc = ms.test(model, validation_loader, criterion, device=device)

        # end_time = time.time()
        #
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')

        # print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.5f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.5f} |  Val. Acc: {valid_acc*100:.2f}%')