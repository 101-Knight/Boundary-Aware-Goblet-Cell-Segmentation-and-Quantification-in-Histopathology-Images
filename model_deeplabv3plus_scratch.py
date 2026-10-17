import segmentation_models_pytorch as smp

def get_model():

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    return model

print("DeepLabV3+ Scratch Model Loaded")