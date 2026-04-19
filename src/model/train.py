import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, save_dir, patience=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.patience = patience
        self.epochs_no_improve = 0
        self.best_loss = float('inf')
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                inputs, targets = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
            val_loss = self.validate()
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            logging.info(f'Validation Loss: {val_loss:.4f}')
            self.checkpoint(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    logging.info('Early stopping!')
                    break

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def checkpoint(self, val_loss):
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        logging.info(f'Checkpoint saved at {checkpoint_path}')  

    def close(self):
        self.writer.close()
