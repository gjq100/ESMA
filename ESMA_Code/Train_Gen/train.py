import os
from typing import Dict
import numpy as np
import torchvision
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from .Generator import Generator
from .Trainer import GeneratorTrainer
import torch.nn as nn
import torch.nn.functional as F
import time
from .Sample_screening import Sample_screening
import timm

def pretrained_model(model_name,pretrained=True,device='cuda:0'):

    if model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=pretrained)
    elif model_name == "vgg19bn":
        model = torchvision.models.vgg19_bn(pretrained=pretrained)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=pretrained)
    elif model_name == "incres_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=pretrained)
    elif model_name == "ens_adv_inception_resnet_v2":
        model = timm.create_model("ens_adv_inception_resnet_v2", pretrained=pretrained)
    elif model_name == "inception_v3":
        model = timm.create_model("inception_v3", pretrained=pretrained)
    elif model_name == "inception_v4":
        model = timm.create_model("inception_v4", pretrained=pretrained)
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

def feature_extraction(model,train_set,device,targets):
    model.eval()
    features = [[] for _ in range(len(targets))]
    with torch.no_grad(): 
        
        for i,(inputs,labels) in enumerate(train_set):
            inputs,labels=inputs.to(device),labels.to(device)
            output=model(normalize(inputs))
            feature = output
            for j in range(labels.shape[0]):
                features[targets.index(int(labels[j]))].append(feature[j].squeeze(-1).squeeze(-1))

    for i in range(len(features)):
        features[i] = torch.mean(torch.stack(features[i]),dim=0)
        features[i] = np.array(features[i].to('cpu'))
    
    features = np.array(features)
    return(features)

def double_kl_div(p_output, q_output):
    KLDivLoss = nn.KLDivLoss()
    return (KLDivLoss(F.log_softmax(p_output,dim=-1), F.softmax(q_output,dim=-1)) + KLDivLoss(F.log_softmax(q_output,dim=-1), F.softmax(p_output,dim=-1)))

def train(modelConfig: Dict):
    time_start = time.time()
    device = torch.device(modelConfig["device"])

    scale_size = 256
    img_size = 224
    src = 'imagenet_train'
    source_model = pretrained_model(modelConfig["Source_Model"], pretrained=True, device=device).to(device)
    source_model.load_state_dict(torch.load("models/"+modelConfig["Source_Model"]))
    
    source_model.eval()
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.03, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    ])
    train_match_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
    ])
    train_set = torchvision.datasets.ImageFolder(src, train_transform)
    train_set_match = torchvision.datasets.ImageFolder(src, train_match_transform)
    targets = [24,99,245,344,471,555,661,701,802,919]
    source_samples = []
    for img_name, label in train_set.samples:
        if label in targets:
            source_samples.append((img_name, label))
    train_set.samples = source_samples  
    source_samples_match = []
    for img_name, label in train_set_match.samples:
        if label in targets:
            source_samples_match.append((img_name, label))
    train_set_match.samples = source_samples_match
    contain_embeddings = ['downblocks','middleblocks','upblocks']

    generator = Generator( num_target=len(targets), ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        Screening = Sample_screening(source_model, train_set_match, device, targets, q=modelConfig["q"])
        target_match = Screening.find_target_matches()

        dataloader = DataLoader(
        train_set, batch_size=modelConfig["batch_size"], shuffle=True,num_workers=12, pin_memory=True)
        generator.load_state_dict(torch.load(os.path.join(
            modelConfig["Generator_save_dir"], modelConfig["training_load_weight"]), map_location=device),strict=False)
        print("Model weight load down.")
        for name, parameter in generator.target_embedding.named_parameters():
            parameter.requires_grad = False
        for m in generator._modules.items():
            if  m[0] in contain_embeddings:
                for i,layers in enumerate(m[1]):
                    if 'Sample' not in type(layers).__name__ :
                        for name, parameter in layers.target_proj.named_parameters():
                            parameter.requires_grad = False
                
    else:
        source_samples = []
        for img_name, label in train_set_match.samples:
            if label in targets:
                source_samples.append((img_name, label))
        train_set_match.samples = source_samples 
        dataloader = DataLoader(
        train_set_match, batch_size=modelConfig["batch_size"], shuffle=True,num_workers=12, pin_memory=True)
        generator.weight_init()
        for name, parameter in generator.named_parameters():
            parameter.requires_grad = False
        for name, parameter in generator.target_embedding.named_parameters():
            parameter.requires_grad = True
        for m in generator._modules.items():
            if  m[0] in contain_embeddings:
                for i,layers in enumerate(m[1]):
                    if 'Sample' not in type(layers).__name__ :
                        for name, parameter in layers.target_proj.named_parameters():
                            parameter.requires_grad = True
        features = torch.tensor(feature_extraction(source_model,dataloader,device,targets)).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()), lr=modelConfig["lr"],weight_decay=5e-5)

    try:
        features
    except NameError:
        features_exist = False
    else:
        features_exist = True
    if features_exist:

        optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()), lr=2e-5,weight_decay=5e-5)
        
        for e in range(modelConfig["epoch"]):
            emb = generator.target_embedding(torch.tensor([0,1,2,3,4,5,6,7,8,9],device=device))
            distance_matrix = torch.norm(emb[:,None]-emb,dim=2,p=2)

            features_distances = torch.norm(features[:,None]-features,dim=2,p=2)
            cosine_matrix = torch.matmul(F.normalize(emb,dim=-1),F.normalize(emb,dim=-1).T)

            features_cosine = torch.matmul(F.normalize(features,dim=-1),F.normalize(features,dim=-1).T)
            kl_loss = double_kl_div(features_distances,distance_matrix)+5*double_kl_div(features_cosine,cosine_matrix)
            sum_loss = 0.01*torch.norm(emb)
            for m in generator._modules.items():
                if  m[0] in contain_embeddings:
                    for i,layers in enumerate(m[1]):
                        if 'Sample' not in type(layers).__name__ :
                            proj_emb = layers.target_proj(emb)
                            distance_matrix = torch.norm(proj_emb[:,None]-proj_emb,dim=2,p=2)
                            cosine_matrix = torch.matmul(F.normalize(proj_emb,dim=-1),F.normalize(proj_emb,dim=-1).T)
                            kl_loss += double_kl_div(features_distances,distance_matrix)+5*double_kl_div(features_cosine,cosine_matrix)
                            sum_loss += 0.01*torch.norm(proj_emb)
            sum_loss += kl_loss
            sum_loss.backward()
            if e%1000 == 0:
                print('%d th epoch: loss is %2f'%(e,kl_loss.item()))
            optimizer.step()

            torch.cuda.empty_cache()
        torch.save(generator.state_dict(), os.path.join(
        modelConfig["Generator_save_dir"], 'ckpt_' + "pretrained_" +modelConfig["Source_Model"] + "_.pt"))

        time_end = time.time()
        print('time cost'+':  ',time_end-time_start,'s')
        quit()

    trainer = GeneratorTrainer(
        model=generator, source_model=source_model, target_match=target_match, num_target=len(targets), eps=10/255).to(device)
    
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:

                optimizer.zero_grad()
                x_0 = images.to(device)
                for i in range(labels.shape[0]):
                    labels[i] = targets.index(labels[i].item())
                labels = labels.to(device)
                loss = trainer(x_0,src_labels=labels)
                loss.backward()

                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item()
                })
        torch.cuda.empty_cache()
        if (e+1)%10==0:
            torch.save(generator.state_dict(), os.path.join(
            modelConfig["Generator_save_dir"], 'ckpt_' + str(e) + "_" + modelConfig["Source_Model"] +"_.pt"))
            
    time_end = time.time()
    print('time cost'+':  ',time_end-time_start,'s')

