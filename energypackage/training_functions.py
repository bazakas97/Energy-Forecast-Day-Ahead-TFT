import os
import math
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#  Definition of Quantile Loss (Pinball Loss)

class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        target_expanded = target.expand_as(preds)
        errors = target_expanded - preds
        losses = []
        for i, q in enumerate(self.quantiles):
            error = errors[..., i]
            loss_q = torch.max(q * error, (q - 1) * error)
            losses.append(loss_q.unsqueeze(-1))
        losses = torch.cat(losses, dim=-1)
        return losses.mean()

def train_tft_model(model, train_loader, val_loader, epochs, output_dir, optimizer_lr=0.0001, quantiles=[0.1, 0.5, 0.9]):
    os.makedirs(output_dir, exist_ok=True)
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
    device = next(model.parameters()).device
    best_r2 = -math.inf

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_t = X_batch.to(device)
            y_t = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_t = X_batch.to(device)
                y_t = y_batch.to(device)
                preds = model(X_t)
                # Χρήση της μεσικής πρόβλεψης (50ο percentil, index 1)
                median_preds = preds[..., 1]
                all_preds.append(median_preds.cpu().numpy())
                all_targets.append(y_t.cpu().numpy())
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        val_mae = mean_absolute_error(all_targets.ravel(), all_preds.ravel())
        val_rmse = math.sqrt(mean_squared_error(all_targets.ravel(), all_preds.ravel()))
        val_r2 = r2_score(all_targets.ravel(), all_preds.ravel())
        print(f"TFT Epoch {epoch+1}/{epochs}, Loss={train_loss:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R2={val_r2:.4f}")
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    return
