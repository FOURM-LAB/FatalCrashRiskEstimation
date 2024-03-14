from torchvision import transforms

def base_aug(image_size=768):
    transform_basic = transforms.Compose([   
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), #recommended to do it before normalizing    
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])     
    return transform_basic    


def pretrain_aug(image_size=768):
    transform_aug = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                               saturation=0.4, hue=0.1),    
        transforms.ToTensor(), #recommended to do it before normalizing    
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])    
    return transform_aug