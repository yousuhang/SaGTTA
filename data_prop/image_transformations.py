import albumentations as A


def get_transform(phase):
    if phase == 'train':
        return [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(p=0.5),
            A.GaussianBlur(),
            A.RandomBrightnessContrast(0.2,0.2, always_apply=True),
            A.RandomGamma(),
            A.RandomResizedCrop(256, 256, scale=(0.5, 1.0)),
        ]
    elif phase == 'test' or phase == 'val':
        return [A.Resize(256, 256)]

