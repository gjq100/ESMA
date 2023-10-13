# ESMA
Pytorch implementation of paper "Perturbation Towards Easy Samples Improves Targeted Adversarial Transferability", Junqi Gao, Biqing Qi, Yao Li, Zhichang Guo, Dong Li, Yuming Xing, Dazhi Zhang.
The following prerequisites we used:
- Python (version 3.11.4)
- Pytorch (version 1.11.0+cu113)
- torchvision (version 0.12.0+cu113)
- timm (version 0.9.2)
### Step 1: Pretrain Embedding

To pretrain the embeddings using a specific source model (e.g., ResNet50), run the following command:

'''
python ESMA.py --batch_size 25 --Source_Model ResNet50 --state pretrain_embedding
'''

### Step 2: Train ESMA

To train the ESMA model using the pre-trained embedding, run the following command:

'''
python ESMA.py --batch_size 25 --Source_Model ResNet50 --epoch 300 --training_load_weight ckpt_pretrained_ResNet50_.pt --q 2 --state train_model
'''

### Step 3: Adversarial Testing

To perform adversarial testing using the trained ESMA model, run the following command:

'''
python ESMA.py --batch_size 25 --Source_Model ResNet50 --test_load_weight ckpt_299_ResNet50_.pt --state advtest
'''
