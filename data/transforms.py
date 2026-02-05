import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

def get_train_transforms(img_size):
    """获取训练时的数据增强变换
    Args:
        img_size (tuple): 图像尺寸 (width, height)
    """
    if isinstance(img_size, str):
        img_size = tuple(map(int, img_size.split(',')))
    elif isinstance(img_size, list):
        img_size = tuple(img_size)
    
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_val_transforms(img_size):
    """获取验证时的数据变换
    Args:
        img_size (tuple): 图像尺寸 (width, height)
    """
    if isinstance(img_size, str):
        img_size = tuple(map(int, img_size.split(',')))
    elif isinstance(img_size, list):
        img_size = tuple(img_size)
    
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])