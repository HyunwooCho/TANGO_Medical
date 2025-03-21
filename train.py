from monai.networks.nets import SwinUNETR
import torch.optim as optim

# 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=1, feature_size=48).to(device)

# 손실 함수 및 최적화기
loss_function = monai.losses.DiceLoss(sigmoid=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
