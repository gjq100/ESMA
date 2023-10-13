import os
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from .Generator import Generator
from .gaussian_smoothing import get_gaussian_kernel
import timm

def pretrained_model(model_name, pretrained=True):

    if model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True)
    elif model_name == "vgg19bn":
        model = torchvision.models.vgg19_bn(pretrained=True)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == "incres_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    elif model_name == "ens_adv_inception_resnet_v2":
        model = timm.create_model("ens_adv_inception_resnet_v2", pretrained=True)
    elif model_name == "inception_v3":
        model = timm.create_model("inception_v3", pretrained=True)
    elif model_name == "inception_v4":
        model = timm.create_model("inception_v4", pretrained=True)
    elif model_name == "adv_inc_v3":
        model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=True)
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)
    else:
        raise ValueError(f"Not supported model name. {model_name}")
    return model


def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t

def advtest(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    scale_size = 256
    img_size = 224
    src = 'imagenet_val'
    val_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
    ])
    val_set = torchvision.datasets.ImageFolder(src, transform=val_transform)
    targets = [24,99,245,344,471,555,661,701,802,919]
    source_samples = []

    for img_name, label in val_set.samples:
        
        if label in targets:
            source_samples.append((img_name, label))
    val_set.samples = source_samples
    generator = Generator( num_target=len(targets), ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                        num_res_blocks=modelConfig["num_res_blocks"]).to(device)
    ckpt = torch.load(os.path.join(
                modelConfig["Generator_save_dir"], modelConfig["test_load_weight"]), map_location=device)
    generator.load_state_dict(ckpt,strict=False)
    print("model load weight done.")
    generator.eval()
    modellist = ["ResNet50","ResNet152","vgg19bn",'DenseNet121',"incres_v2","inception_v3","inception_v4", "adv_inc_v3", "ens_adv_inception_resnet_v2", "vit" ]
    target_model_list = [model for model in modellist if model != modelConfig["Source_Model"]]
    for name in target_model_list:
        target_model = pretrained_model(name,pretrained=True).to(device)
        target_model.load_state_dict(torch.load("models/"+name))
        target_model.eval()
        targeted_attack(generator,name,target_model,eps=16/255,val_set=val_set,targets=targets,device=modelConfig["device"])

def attack(generator,target_model,target,eps,val_set,targets,device):
    target_label = target
    source_samples = []
    for img_name, label in val_set:
        if label!=target_label:
            source_samples.append((img_name, label))
    source_set = DataLoader(source_samples, batch_size=50, shuffle=False, num_workers=12,
                                           pin_memory=True)
    with torch.no_grad():
        total = 0
        acc = 0
        total_batch = 0
        acc_batch = 0

        for imgs,labels in source_set:
            imgs = imgs.to(device)
            for i in range(labels.shape[0]):
                labels[i] = targets.index(labels[i].item())
            label = labels.to(device)
            kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).cuda()
            labels = (targets.index(target_label)*torch.ones((imgs.shape[0],))).long().to(device)
            perturbated_imgs = generator(imgs,  target=labels)
            perturbated_imgs = kernel(perturbated_imgs)
            attx = torch.min(torch.max(perturbated_imgs, imgs-eps), imgs + eps)
            attx = torch.clamp(attx, 0, 1.0)
            out = target_model(normalize(attx.clone().detach()))
            acc += torch.sum(torch.eq(out.argmax(dim=-1),target_label)).item()
            total += imgs.shape[0]
            acc_batch += torch.sum(torch.eq(out.argmax(dim=-1),target_label)).item()
            total_batch += imgs.shape[0]    
            acc_batch = 0
            total_batch = 0
    return(acc/total)
def targeted_attack(generator,name,target_model,eps,val_set,targets,device):
    acc = 0.0
    for target in targets:
        acc += attack(generator,target_model,target,eps = eps,val_set=val_set,targets=targets,device=device)
    print('average rate for '+name+':',acc/len(targets))
