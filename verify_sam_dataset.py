import torch

from model_unetpp_features import UNetPPFeatures

dummy = torch.randn(
    1,
    3,
    256,
    256
)

model = UNetPPFeatures()

with torch.no_grad():
    y = model(dummy)

print("Output Shape:", y.shape)