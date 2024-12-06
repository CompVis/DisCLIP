import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
from torch.nn import functional as F
import clip
import tqdm
from collections import OrderedDict
from utils import generate_embed_ds,load_embedding_datasets,load_json, get_text_encoding_tensor_from_list, get_global_images_and_labels, save_selected_descriptions_imagewise, get_imagewise_cls_description_texts_from_mask_tensor, get_cls_description_embeddings_tensor, get_classwise_cls_description_texts_from_mask_tensor,load_vision_language_model
from method import get_top_k_ambiguous_classes_0s_per_image, get_selection_masks_from_vlm_feedback_imagewise, eval_cls_ful_descriptions_imagewise, eval_cls_ful_des_plus_cls_less_des_imagewise, eval_cls_ful_des_plus_cls_less_des_classwise, eval_cls_ful_descriptions_classwise
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="choose dataset from following strings: ['flowers','dtd','places','eurosat','food','pets','ilsvrc','imagenet_v2']", default='flowers')
    parser.add_argument('--pool', type=str, help="choose description pool from following strings: ['dclip','con_llama']", default='dclip')
    parser.add_argument('--encoding_device', type=int, help="Cuda ID to encode images and texts", default=0)
    parser.add_argument('--calculation_device', type=int, help="Cuda ID to perform evaluation", default=1)
    parser.add_argument('--k_ambiguous_imagewise_classes', type=int, help="Amount of ambiguous classes to consider per iamge", default=3)
    parser.add_argument('--m_relevant_descriptions', type=int, help="Amount of descriptions to select per ambiguous class", default=5)
    parser.add_argument('--n_reference_samples', type=int, help="Number of reference samples used to construct S. Value is downsized to the cardinality of the smallest training class.", default=1000)
    parser.add_argument('--batch_size', type=int, help="For encoding and other batched operations", default=1000)
    parser.add_argument('--descriptions_save_path', type=str, help="Relative path to store selected descriptions", default='./saved_descriptions')
    parser.add_argument('--eval_path', type=str, help="Relative path to store evaluation results", default='./eval')
    parser.add_argument('--cls_weight_range', type=str, help="Range of weights to evaluate in classnamefree mode", default='np.arange(0, 40, 0.25)')
    args = parser.parse_args()
    #modify other parameters as needed. If n_refrence_samples > smallest_train_class_cardinality it will be downscaled automatically.
    run_params = vars(args)
    
    assert run_params['dataset'] in ['flowers','dtd','places','eurosat','food','pets','cub','ilsvrc','imagenet_v2']
    assert run_params['pool'] in ['dclip','con_llama']
    
    ##setup evaluation save paths and datasets##
    if not os.path.exists(run_params['eval_path']):
        os.mkdir(run_params['eval_path'])
    i=0
    while os.path.exists(os.path.join(run_params['eval_path'],f'{run_params["dataset"]}_{run_params["pool"]}_run_{i}')):
        i+=1
    eval_path = os.path.join(run_params['eval_path'],f'{run_params["dataset"]}_{run_params["pool"]}_run_{i}')
    os.mkdir(eval_path)
    run_params['eval_path']=eval_path

    if not os.path.exists(os.path.join('.','image_embeddings',f'{run_params["dataset"]}_embeds')):
        generate_embed_ds(run_params,run_params['calculation_device'],run_params['batch_size'])

    if not os.path.exists(run_params['descriptions_save_path']):
        os.mkdir(run_params['descriptions_save_path'])
    

    selection_dataset, eval_dataset = load_embedding_datasets(run_params)

    selection_dataloader = DataLoader(selection_dataset, run_params['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, run_params['batch_size'], shuffle=False,num_workers=8, pin_memory=True)
    dataloaders = {'selection_dataloader':selection_dataloader,'eval_dataloader':eval_dataloader}


    class_language_data = load_json(os.path.join('.','descriptions',f'descriptions_{run_params["dataset"]}_{run_params["pool"]}.json'))
    fallback_class_language_data = load_json(os.path.join('.','descriptions',f'descriptions_{run_params["dataset"]}_dclip.json'))

    def sentence_pattern_cls(class_name):
        return f'A photo of a {class_name}.'

    def sentence_pattern_cls_plus_des(class_name,description):
        return f'A photo of a {class_name}, {description}.'

    class_indices_str = list(class_language_data["index_to_classname"].keys())
    class_indices_tensor = torch.tensor([int(idx) for idx in class_indices_str])
    index_to_classname = class_language_data["index_to_classname"]
    classname_texts = [index_to_classname[index] for index in class_indices_str]
    captions_texts = list(sentence_pattern_cls(index_to_classname[index]) for index in class_indices_str)
    index_to_descriptions = {index: class_language_data["index_to_descriptions"][index] if class_language_data["index_to_descriptions"][index] != [] else fallback_class_language_data["index_to_descriptions"][index] for index in class_indices_str}

    description_texts_lol = sorted(list(index_to_descriptions.values()))
    description_texts_gs = list(OrderedDict.fromkeys([description for description_list in description_texts_lol for description in description_list]))

    LLM_assignment_masks_acc = {}
    random_assignment_masks_acc = {}
    for classlabel in tqdm.tqdm(index_to_descriptions.keys()):
        descriptions = index_to_descriptions[classlabel]
        selection_mask = torch.zeros(len(description_texts_gs),dtype=torch.float16)
        random_mask = torch.zeros(len(description_texts_gs),dtype=torch.float16)
        sel_index_acc = []
        rand_index_acc = []
        for description in descriptions:
            sel_index_acc.append(description_texts_gs.index(description))
            rand_index_acc.append(np.random.randint(0,len(description_texts_gs)))
        selection_mask[np.array(sel_index_acc)]=1
        random_mask[np.array(rand_index_acc)]=1
        LLM_assignment_masks_acc[classlabel] = selection_mask.to(run_params['calculation_device'])
        random_assignment_masks_acc[classlabel] = random_mask.to(run_params['calculation_device'])


    LLM_assignment_mask = LLM_assignment_masks_acc
    LLM_mask_tensor = torch.cat([torch.eye(len(class_indices_tensor),dtype=torch.float16,device=run_params['calculation_device']),torch.stack(list(LLM_assignment_mask.values()))],dim=1)
    random_assignment_mask = random_assignment_masks_acc
    random_mask_tensor = torch.cat([torch.eye(len(class_indices_tensor),dtype=torch.float16,device=run_params['calculation_device']),torch.stack(list(random_assignment_mask.values()))],dim=1)

    vlm, preprocess = load_vision_language_model(run_params)
    caption_encodings = get_text_encoding_tensor_from_list(vlm,captions_texts,run_params['encoding_device'],run_params['batch_size'])
    description_encodings = get_text_encoding_tensor_from_list(vlm,description_texts_gs,run_params['encoding_device'],run_params['batch_size'])
    caption_encodings = caption_encodings.to(run_params['calculation_device'])
    description_encodings = description_encodings.to(run_params['calculation_device'])

    global_selection_image_encodings, global_selection_labels = get_global_images_and_labels(run_params,selection_dataloader)
    #assert that global selection labels are in ascending order
    assert torch.all(global_selection_labels[1:] >= global_selection_labels[:-1])
    global_eval_image_encodings, global_eval_labels = get_global_images_and_labels(run_params,eval_dataloader)
    ############################################################################################################

    top_k_ambiguous_classes_per_image = get_top_k_ambiguous_classes_0s_per_image(global_eval_image_encodings,global_eval_labels,caption_encodings,run_params)

    print('Getting description selections.')
    top_ambiguous_selection_masks_vlm_feedback = get_selection_masks_from_vlm_feedback_imagewise(global_selection_image_encodings,global_eval_labels,description_encodings,top_k_ambiguous_classes_per_image,class_indices_tensor,run_params)
    #fill up full 0 selection_masks with LLM_assignment_mask. This is to make the evaluation more fair and balanced
    for i,image_encoding in enumerate(global_eval_image_encodings):
        for j in range(run_params['k_ambiguous_imagewise_classes']):
            if torch.all(top_ambiguous_selection_masks_vlm_feedback[i,j,len(caption_encodings):] == 0):
                LLM_assignment_mask_raw = LLM_assignment_mask[str(top_k_ambiguous_classes_per_image[i,j].item())].clone()
                indices = LLM_assignment_mask_raw.nonzero().squeeze(1)
                LLM_assignment_mask_raw[indices[run_params['m_relevant_descriptions']:]]=0
                top_ambiguous_selection_masks_vlm_feedback[i,j,len(caption_encodings):] = LLM_assignment_mask_raw

    print('Saving selected descriptions.')
    save_selected_descriptions_imagewise(top_ambiguous_selection_masks_vlm_feedback,run_params,description_texts_gs,index_to_classname,top_k_ambiguous_classes_per_image)

    print('Evaluating description selection in mode: classname-free')
    eval_cls_ful_des_plus_cls_less_des_imagewise(caption_encodings,description_encodings,top_ambiguous_selection_masks_vlm_feedback,global_eval_image_encodings,global_eval_labels,run_params,class_indices_str,'selected',top_k_ambiguous_classes_per_image)

    print('Getting classname-containing texts.')
    selection_heuristic_cls_description_texts_dict = get_imagewise_cls_description_texts_from_mask_tensor(top_ambiguous_selection_masks_vlm_feedback,sentence_pattern_cls_plus_des,index_to_classname,top_k_ambiguous_classes_per_image,description_texts_gs,run_params)

    print('Getting classname-containing embeddings.')
    selection_heuristic_cls_description_embeddings_tensor = get_cls_description_embeddings_tensor(vlm,selection_heuristic_cls_description_texts_dict,run_params)

    print('Evaluating description selection in mode: classname-containing')
    eval_cls_ful_descriptions_imagewise(selection_heuristic_cls_description_embeddings_tensor,global_eval_image_encodings,global_eval_labels,run_params,class_indices_str,top_k_ambiguous_classes_per_image)

    eval_cls_ful_des_plus_cls_less_des_classwise(caption_encodings,description_encodings,LLM_assignment_mask,class_indices_str,run_params,'LLM_assignment',global_eval_image_encodings,global_eval_labels)
    eval_cls_ful_des_plus_cls_less_des_classwise(caption_encodings,description_encodings,random_assignment_mask,class_indices_str,run_params,'random_assignment',global_eval_image_encodings,global_eval_labels)

    #ToDO: Put instructive print statements here, or use tqdm etc
    LLM_cls_description_texts_dict = get_classwise_cls_description_texts_from_mask_tensor(LLM_assignment_masks_acc,sentence_pattern_cls_plus_des,index_to_classname,description_texts_gs)
    random_cls_description_texts_dict = get_classwise_cls_description_texts_from_mask_tensor(random_assignment_masks_acc,sentence_pattern_cls_plus_des,index_to_classname,description_texts_gs)
    LLM_cls_description_embeddings_dict = {key: get_text_encoding_tensor_from_list(vlm,value,run_params['encoding_device'],run_params['batch_size']).to(run_params['calculation_device']) for key,value in tqdm.tqdm(LLM_cls_description_texts_dict.items())}
    random_cls_description_embeddings_dict = {key: get_text_encoding_tensor_from_list(vlm,value,run_params['encoding_device'],run_params['batch_size']).to(run_params['calculation_device']) for key,value in tqdm.tqdm(random_cls_description_texts_dict.items())}

    eval_cls_ful_descriptions_classwise(LLM_cls_description_embeddings_dict,global_eval_image_encodings,global_eval_labels,run_params,class_indices_str,'LLM_assignment')
    eval_cls_ful_descriptions_classwise(random_cls_description_embeddings_dict,global_eval_image_encodings,global_eval_labels,run_params,class_indices_str,'random_assignment')