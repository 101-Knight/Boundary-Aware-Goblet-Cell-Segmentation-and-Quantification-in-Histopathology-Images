import segmentation_models_pytorch as smp

def get_model():

    model = smp.DeepLabV3(
        encoder_name="resnet34",
        encoder_weights="imagenet",   # PRETRAINED
        in_channels=3,
        classes=1
    )

    return model

print("Pretrained DeepLabV3 Loaded Successfully")