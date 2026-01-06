import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_data_loaders
from models.model import DepthNet

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader = get_data_loaders('data', pseudo_dir='pseudo_data', batch_size=4)

# Initialize model
model = DepthNet().to(device)
criterion = nn.L1Loss()  # MAE loss works well for depth
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for imgs, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, depths = imgs.to(device), depths.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, depths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, depths in val_loader:
            imgs, depths = imgs.to(device), depths.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, depths)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"checkpoints/depthnet_epoch{epoch+1}.pth")

print("✅ Training Complete!")
