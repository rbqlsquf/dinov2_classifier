# 노트북에서 사용할 수정된 셀 코드
import time

# Configuration
batch_size = 1024
num_epochs = 2
learning_rate = 0.001
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_module = CustomCIFAR10DataModule(batch_size=batch_size, transform=transform)
model = CustomModel(embed_dim=dinov2_vits14.embed_dim, num_classes=10, learning_rate=learning_rate)
trainer = pl.Trainer(
    max_epochs=num_epochs,
    check_val_every_n_epoch=1,
)

# 시간 측정 시작
start_time = time.time()

# 모델 훈련 및 검증
trainer.fit(model, data_module)
trainer.validate(model, datamodule=data_module)

# 시간 측정 종료
end_time = time.time()

print(f"Training and testing complete. Total time: {end_time - start_time:.2f} seconds") 