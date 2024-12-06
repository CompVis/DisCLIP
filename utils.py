import os
import clip
import torch
import numpy as np
from tqdm import tqdm
from cub_200.cub200 import CUBDataset
from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision.datasets import Flowers102, Places365, DTD, EuroSAT, Food101, OxfordIIITPet, ImageNet
import torch.nn.functional as F
from datetime import datetime
import yaml

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from torch.utils.data import Dataset
import numpy as np
import torch
import os


def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)

def get_selection_indices(labels,selection_amount):
    counter_array = np.zeros(len(np.unique(labels)))
    selection_set_indices = []
    train_set_indices = []
    for idx, label in enumerate(labels):
        if counter_array[label] < selection_amount:
            counter_array[label] += 1
            selection_set_indices.append(idx)
        else:
            train_set_indices.append(idx)
    minimal_selection_amount = min(counter_array)
    if minimal_selection_amount < selection_amount:
        print(f"Selection amount needs to be reduced to {minimal_selection_amount} due to the dataset size")
        return selection_set_indices, train_set_indices, (True, minimal_selection_amount)
    else:
        return selection_set_indices, train_set_indices, (False, selection_amount)


class Embedding_Dataset(Dataset):
    def __init__(self, root_dir,run_params,selection_amount=None,indices=None):
        self.embeddings = np.load(os.path.join(root_dir, '_features.npy'))
        self.labels = np.load(os.path.join(root_dir, '_labels.npy'))
        if indices is not None:
            self.embeddings = self.embeddings[indices]
            self.labels = self.labels[indices]
        assert len(self.embeddings) == len(self.labels)
        if selection_amount is not None:
            selection_set_indices, train_set_indices, (retry, reduced_selection_amount) = get_selection_indices(self.labels,selection_amount)
            if retry:
                selection_set_indices, train_set_indices, (retry, selection_amount) = get_selection_indices(self.labels,reduced_selection_amount)
                assert(reduced_selection_amount==selection_amount)
                run_params['n_reference_samples']=int(selection_amount)
                print(f"Selection amount was reduced to {reduced_selection_amount} due to the dataset size")
                if retry:
                    raise ValueError("Something went wrong. Please fix.")
            self.embeddings = self.embeddings[selection_set_indices]
            self.labels = self.labels[selection_set_indices]
            self.train_set_indices = train_set_indices
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float16)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def create_embed_ds_0(run_params,ds,split,device,batch_size):
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()
    save_path = os.path.join('.','image_embeddings',f'{run_params["dataset"]}_embeds/{split}')
    os.makedirs(save_path,exist_ok=True)

    features = []
    labels = []
    batch_acc = []
    for idx, (image, label) in enumerate(tqdm(ds,desc='Encoding images: iterating through dataset')):

        image = preprocess(image).unsqueeze(0)
        batch_acc.append(image)
        labels.append(label)

        if ((idx%batch_size) and (idx!=0)) or (idx==len(ds)-1):
            with torch.no_grad():
                batch_acc_tensor = torch.cat(batch_acc,0).to(device) 
                image_features = model.encode_image(batch_acc_tensor)
            features.append(image_features.cpu().numpy())
            batch_acc = []

        
    features = np.concatenate(features,0)
    labels = np.array(labels)
    np.save(os.path.join(save_path,'_features.npy'), features)
    np.save(os.path.join(save_path,'_labels.npy'), labels)
    del model
    torch.cuda.empty_cache()


def load_embedding_datasets(run_params):
    embed_ds_dir = os.path.join('.','image_embeddings',f'{run_params["dataset"]}_embeds')
    embed_ds_dir_train = os.path.join(embed_ds_dir,'train')
    embed_ds_dir_test = os.path.join(embed_ds_dir,'test')
    
    if run_params['dataset']=='eurosat':
        embed_ds_dir_test = embed_ds_dir_train
        selection_dataset = Embedding_Dataset(embed_ds_dir_train,run_params,selection_amount=run_params['n_reference_samples'])
        eval_dataset = Embedding_Dataset(embed_ds_dir_test,run_params,selection_amount=None,indices=selection_dataset.train_set_indices)
        return selection_dataset, eval_dataset
    
    selection_dataset = Embedding_Dataset(embed_ds_dir_train,run_params,selection_amount=run_params['n_reference_samples'])
    eval_dataset = Embedding_Dataset(embed_ds_dir_test,run_params)
    return selection_dataset, eval_dataset


def generate_embed_ds(run_params,device,batch_size):
    root = os.path.join('.','datasets')
    if not os.path.exists(root):
        os.mkdir(root)
    if run_params['dataset']=='cub':
        root = os.path.join(root,'cub_200','CUB_200_2011')
        ds_train, ds_test = CUBDataset(root,train=True),CUBDataset(root,train=False)
    if run_params['dataset']=='ilsvrc':
        root = os.path.join(root,'ilsvrc')
        ds_train, ds_test = ImageNet(root=root,split='train'),ImageNet(root=root,split='val')
    if run_params['dataset']=='imagenet_v2':
        train_root = os.path.join(root,'ilsvrc')
        test_root = os.path.join(root,'imagenet_v2')
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        ds_train, ds_test = ImageNet(root=train_root,split='train'),ImageNetV2Dataset(location=test_root)
    if run_params['dataset']=='flowers':
        ds_train, ds_test = Flowers102(root=root,split='train',download=True),Flowers102(root=root,split='test',download=True)
    if run_params['dataset']=='places':
        ds_train, ds_test = Places365(root=root,split='train-standard',small=True,download=True),Places365(root=root,split='val',small=True,download=True)
    if run_params['dataset']=='dtd':
        ds_train, ds_test = DTD(root=root,split='train',download=True),DTD(root=root,split='test',download=True)
    if run_params['dataset']=='eurosat':
        ds_train, ds_test = EuroSAT(root=root,download=True),EuroSAT(root=root,download=True)
    if run_params['dataset']=='food':
        ds_train, ds_test = Food101(root=root,split='train',download=True),Food101(root=root,split='test',download=True)
    if run_params['dataset']=='pets':
        ds_train, ds_test = OxfordIIITPet(root=root,split='trainval',download=True),OxfordIIITPet(root=root,split='test',download=True)

    datasets = {'train':ds_train,'test':ds_test}
    if run_params['dataset'] == 'eurosat':
        datasets = {'train':ds_train}
    
    if run_params['dataset'] == 'ilsvrc':
        imagenet_v2_train_path = os.path.join('.','image_embeddings','imagenet_v2_embeds','train')
        #create symlink of imagenet training ds embeddings if already embedded for ImageNetV2
        if os.path.exists(imagenet_v2_train_path):
            datasets = {'test':ds_test}
            imagenet_path = os.path.join('.','image_embeddings','ilsvrc_embeds')
            if not os.path.exists(imagenet_path):
                os.makedirs(imagenet_path)
            imagenet_train_path = os.path.join(imagenet_path,'train')
            if not os.path.exists(imagenet_train_path):
                os.symlink(os.path.abspath(imagenet_v2_train_path),os.path.abspath(imagenet_train_path),target_is_directory = True)

    if run_params['dataset'] == 'imagenet_v2':
        imagenet_train_path = os.path.join('.','image_embeddings','ilsvrc_embeds','train')
        #create symlink of imagenet training ds embeddings if already embedded for ImageNet
        if os.path.exists(imagenet_train_path):
            datasets = {'test':ds_test}
            imagenet_v2_path = os.path.join('.','image_embeddings','imagenet_v2_embeds')
            if not os.path.exists(imagenet_v2_path):
                os.makedirs(imagenet_v2_path)
            imagenet_v2_train_path = os.path.join(imagenet_v2_path,'train')
            if not os.path.exists(imagenet_v2_train_path):
                os.symlink(os.path.abspath(imagenet_train_path),os.path.abspath(imagenet_v2_train_path),target_is_directory = True)

    for split,ds in tqdm(datasets.items(),desc='Encoding images: iterating through ds partitions'):
        create_embed_ds_0(run_params,ds,split,device,batch_size)

def get_text_encoding_tensor_from_list(model, description_list, device, batch_size):
    num_descriptions = len(description_list)
    tensor_of_token_encodings = []
    for start_idx in range(0, num_descriptions, batch_size):
        end_idx = min(start_idx + batch_size, num_descriptions)
        batch_descriptions = description_list[start_idx:end_idx]
        
        list_of_token_lists = clip.tokenize(batch_descriptions).to(device)
        encodings = model.encode_text(list_of_token_lists)
        tensor_of_token_encodings.append(F.normalize(encodings))
    
    tensor_of_token_encodings = torch.cat(tensor_of_token_encodings, dim=0)
    return tensor_of_token_encodings


def get_global_images_and_labels(run_params,selection_dataloader):
    images_acc = []
    labels_acc = []

    for batch in selection_dataloader:
        image_features, labels = batch
        image_features = image_features
        images_acc.append(F.normalize(image_features))
        labels_acc.append(labels)
    
    #sort labels ascendingly and apply the same sorting to the images
    labels_acc, indices = torch.sort(torch.cat(labels_acc))
    images_acc = torch.cat(images_acc)[indices]

    images_acc = images_acc.to(run_params['calculation_device'])
    labels_acc = labels_acc.to(run_params['calculation_device'])
    return images_acc, labels_acc

def save_selected_descriptions_imagewise(selection_masks_imagewise,run_params,description_texts_gs,index_to_classname,ambiguous_classes_acc):
    time_stamp = datetime.now().strftime("%m-%d %H:%M:%S:%f")
    save_path = os.path.join(run_params['descriptions_save_path'],f'selected_descriptions_{run_params["dataset"]}_{run_params["pool"]}_{time_stamp}.json')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np_des_gs = np.array(description_texts_gs)
    save_dict_0 = {f'image_{i}':{index_to_classname[str(int(cls_idx))]: list(np_des_gs[selection_masks_imagewise[i,j,-len(description_texts_gs):].cpu().bool()]) for j,cls_idx in enumerate(ambiguous_classes)} for i, ambiguous_classes in enumerate(ambiguous_classes_acc)}
    save_dict = {"selected_descriptions":save_dict_0,"index_to_classname":index_to_classname,"run_params":run_params}
    with open(save_path, 'w') as fp:
        json.dump(save_dict, fp, indent=4)
    
def get_imagewise_cls_description_texts_from_mask_tensor(selection_masks_imagewise,sentence_pattern,index_to_classname,ambiguous_classes_acc,description_texts_gs,run_params):
    np_des_gs = np.array(description_texts_gs)
    imagewise_classnameless_descriptions = {f'image_{i}':{index_to_classname[str(int(cls_idx))]: list(np_des_gs[selection_masks_imagewise[i,j,-len(description_texts_gs):].cpu().bool()]) for j,cls_idx in enumerate(ambiguous_classes)} for i, ambiguous_classes in enumerate(ambiguous_classes_acc)}
    for k,v in imagewise_classnameless_descriptions.items():
        for k_0,v_0 in v.items():
            if len(v_0) < run_params['m_relevant_descriptions']:
                difference = run_params['m_relevant_descriptions'] - len(v_0)
                len_old = len(v_0)
                if difference < run_params['m_relevant_descriptions']:
                    for i in range(difference):
                        v_0.append(v_0[i%len_old])
                else:
                    for i in range(difference):
                        v_0.append('')
    imagewise_cls_descriptions = {k:[[sentence_pattern(classname, classnameless_des) for classnameless_des in classnameless_des_list] for classname, classnameless_des_list in v.items()] for k,v in imagewise_classnameless_descriptions.items()}
    for k,v in imagewise_classnameless_descriptions.items():
        for k_0,v_0 in v.items():
            assert(len(v_0)==run_params['m_relevant_descriptions'])
    return imagewise_cls_descriptions


def get_cls_description_embeddings_tensor(vlm,selection_heuristic_cls_description_texts_dict,run_params):
    all_descriptions_flattened = [item for li in list(selection_heuristic_cls_description_texts_dict.values()) for subli in li for item in subli]
    tensor_acc = []
    n_batches = (len(all_descriptions_flattened) // run_params['batch_size']) + 1
    for i in tqdm(range(n_batches)):
        start_index = i*run_params['batch_size']
        end_index = start_index + run_params['batch_size']
        if i != (n_batches - 1):
            tokens = clip.tokenize(all_descriptions_flattened[start_index:end_index]).to(run_params['encoding_device'])
        else:
            if start_index >= len(all_descriptions_flattened):
                break
            tokens = clip.tokenize(all_descriptions_flattened[start_index:]).to(run_params['encoding_device'])
        encodings = vlm.encode_text(tokens)
        encodings_normalized = F.normalize(encodings)
        tensor_acc.append(encodings_normalized)

    return torch.cat(tensor_acc)

def save_eval_data(file_name,eval_logs,run_params):
    time_stamp = datetime.now().strftime("%m-%d %H:%M:%S:%f")
    save_path = os.path.join(run_params['eval_path'],f'eval_logs_{file_name}_{time_stamp}.yml')
    with open(save_path, 'w') as fp:
        yaml.dump(eval_logs, fp, indent=4)


def get_classwise_cls_description_texts_from_mask_tensor(mask_tensor,sentence_pattern,index_to_classname,description_texts_gs):
    cls_description_texts_acc_classwise = {}
    for i, selection_mask in enumerate(mask_tensor.values()):
        cls_description_texts = [sentence_pattern(index_to_classname[str(i)],description_texts_gs[j]) for j in range(len(description_texts_gs)) if selection_mask[j] == 1]
        cls_description_texts_acc_classwise[str(i)] = cls_description_texts
    return cls_description_texts_acc_classwise

def load_vision_language_model(run_params):
    model, preprocess = clip.load('ViT-B/32',device=run_params['encoding_device'])
    model.eval()
    model.requires_grad_(False)
    return model, preprocess