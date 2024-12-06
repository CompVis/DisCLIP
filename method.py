import torch
from tqdm import tqdm
import torchmetrics
from utils import save_eval_data
import numpy as np

def get_top_k_ambiguous_classes_0s_per_image(global_image_encodings,global_labels,caption_encodings,run_params):
    k = run_params['k_ambiguous_imagewise_classes']
    similarity_matrix = torch.matmul(global_image_encodings, caption_encodings.T)
    top_k_predictions = torch.topk(similarity_matrix,k,dim=1).indices

    return top_k_predictions



def get_selection_masks_from_vlm_feedback_imagewise(global_selection_image_encodings,global_eval_labels,description_encodings,top_k_minus_one_ambiguous_classes_acc,class_indices_tensor,run_params):
    
    image_language_similarity_matrix = torch.matmul(global_selection_image_encodings, description_encodings.T)
    language_image_similarity_matrix_class_avg =image_language_similarity_matrix.T.reshape(len(description_encodings),len(class_indices_tensor),run_params['n_reference_samples']).mean(dim=2)

    selection_masks_acc = torch.zeros(len(global_eval_labels),run_params['k_ambiguous_imagewise_classes'],len(class_indices_tensor)+len(description_encodings),dtype=torch.float16,device=run_params['calculation_device'])

    def create_selection_mask(length,true_indices):
        mask = torch.zeros(length,dtype=torch.float16,device=run_params['calculation_device'])
        mask[true_indices] = 1
        return mask

    for i,amb_classes_entry in tqdm(enumerate(top_k_minus_one_ambiguous_classes_acc)):
        image_top_k_masks_acc = torch.zeros(run_params['k_ambiguous_imagewise_classes'],len(class_indices_tensor)+len(description_encodings))

        mean_sim_values_of_ambiguous_classes = language_image_similarity_matrix_class_avg[:,amb_classes_entry]

        for j,class_label in enumerate(amb_classes_entry):
            pseudo_gt_label = class_indices_tensor[class_label]
            mean_sim_values_pseudo_gt_label = language_image_similarity_matrix_class_avg[:,pseudo_gt_label]
    
            differences = mean_sim_values_pseudo_gt_label.unsqueeze(1) - mean_sim_values_of_ambiguous_classes

            all_differences_greater_zero = torch.all(differences >= 0,dim=1)
            
            
            mean_differences = torch.mean(differences,dim=1)
            relevant_mean_differences = mean_differences[all_differences_greater_zero]
            relevant_differences_sorted_values, relevant_differences_sorted_indices = torch.sort(relevant_mean_differences,descending=True)

            index_tracker = torch.arange(len(mean_differences),device=run_params['calculation_device'])
            n_most_relevant_indices = relevant_differences_sorted_indices[:run_params['m_relevant_descriptions']]
            relevant_description_indices = index_tracker[all_differences_greater_zero][n_most_relevant_indices]

            selection_mask = create_selection_mask(len(index_tracker),relevant_description_indices)
            class_mask = create_selection_mask(len(class_indices_tensor),pseudo_gt_label)
            image_top_k_masks_acc[j] = torch.cat((class_mask,selection_mask),dim=0)

        selection_masks_acc[i] = image_top_k_masks_acc
        
    return selection_masks_acc


def eval_cls_ful_descriptions_imagewise(cls_description_encodings,global_eval_image_encodings,global_eval_labels,run_params,class_indices_str,top_k_ambiguous_indices):

    top_k_ambiguous_indices_tensor = torch.tensor(top_k_ambiguous_indices,dtype=torch.long,device=run_params['calculation_device'])
    description_ensembling_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=len(class_indices_str)).to(run_params['calculation_device'])
    description_maxing_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=len(class_indices_str)).to(run_params['calculation_device'])
    
    reshaped = cls_description_encodings.reshape((len(global_eval_image_encodings),run_params['k_ambiguous_imagewise_classes'],run_params['m_relevant_descriptions'],-1)).to(run_params['calculation_device'])
    img_lang_sims = torch.einsum('ab,acdb->acd',global_eval_image_encodings,reshaped)

    ensembled_sims = img_lang_sims.mean(dim=-1)
    ensemble_preds = ensembled_sims.argmax(dim=-1)
    predictions_ensembling = top_k_ambiguous_indices_tensor[torch.arange(len(global_eval_image_encodings)),ensemble_preds.to(run_params['calculation_device'])]
    predictions_ensembling = predictions_ensembling.to(run_params['calculation_device'])

    max_preds,_ = img_lang_sims.max(dim=-1)
    max_preds = max_preds.argmax(dim=-1)
    predictions_maxing = top_k_ambiguous_indices_tensor[torch.arange(len(global_eval_image_encodings)),max_preds.to(run_params['calculation_device'])]
    predictions_maxing = predictions_maxing.to(run_params['calculation_device'])

    description_ensembling_accuracy_metric(predictions_ensembling,global_eval_labels)
    description_maxing_accuracy_metric(predictions_maxing,global_eval_labels)

    eval_logs = {}
    eval_logs["eval_method"] = 'classname-containing descriptions, selected, local'
    eval_logs["cls_weight"] = [float(cls_weight) for cls_weight in eval(run_params['cls_weight_range'])]
    eval_logs["top_1_accuracy, language ensembling"] = [100*description_ensembling_accuracy_metric.compute().item() for _ in range(len(eval_logs["cls_weight"]))]
    eval_logs["top_1_accuracy, language maxing"] = [100*description_maxing_accuracy_metric.compute().item() for _ in range(len(eval_logs["cls_weight"]))]
    eval_logs["run_params"] = run_params

    save_eval_data('classname_containing_selected',eval_logs,run_params)


def eval_cls_ful_des_plus_cls_less_des_imagewise(caption_encodings,description_encodings,selection_masks_vlm_feedback,global_eval_image_encodings,global_eval_labels,run_params,class_indices_str,selection_mask_name,top_k_ambiguous_indices):
    language_encodings = torch.cat([caption_encodings,description_encodings],dim=0)
    global_image_description_similarity_matrix = torch.matmul(global_eval_image_encodings, language_encodings.T)
    description_masks_normalized = selection_masks_vlm_feedback[:,:,len(caption_encodings):]/torch.sum(selection_masks_vlm_feedback[:,:,len(caption_encodings):],dim=2,keepdim=True)

    cls_weight_acc = []
    accuracy_acc = []
    for cls_weight in tqdm(eval(run_params['cls_weight_range'])):
        description_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=len(class_indices_str)).to(run_params['calculation_device'])
        
        cls_masks = selection_masks_vlm_feedback[:,:,:len(caption_encodings)].clone()
        cls_masks[cls_masks==1] = cls_weight
        selection_masks_normalized = torch.cat([cls_masks,description_masks_normalized],dim=2)

        #element-wise multiplication with the selection_mask as a mask
        global_image_selection_similarity_matrix = torch.einsum('ij,ikj->ik',global_image_description_similarity_matrix,selection_masks_normalized)

        description_prediction_indices = torch.argmax(global_image_selection_similarity_matrix,dim=1) 
        #now access the indices of the top k ambiguous classes and get the corresponding class labels
        description_predictions = top_k_ambiguous_indices[torch.arange(top_k_ambiguous_indices.size(0)),description_prediction_indices]

        description_accuracy_metric(description_predictions,global_eval_labels)

        cls_weight_acc.append(float(cls_weight))
        accuracy_acc.append(100*description_accuracy_metric.compute().item())   

    eval_logs = {}
    eval_logs["eval_method"] = '1 classname-containing description + classname-free descriptions, local eval'
    eval_logs["selection_mask_name"] = selection_mask_name
    eval_logs["top_1_accuracy, language ensembling"] = accuracy_acc
    eval_logs["cls_weight"] = cls_weight_acc
    eval_logs["run_params"] = run_params

    save_eval_data(f'classname_free_{selection_mask_name}',eval_logs,run_params)

    caption_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=len(class_indices_str)).to(run_params['calculation_device'])
    global_image_caption_similarity_matrix = torch.matmul(global_eval_image_encodings, caption_encodings.T)
    caption_accuracy_metric(global_image_caption_similarity_matrix,global_eval_labels)

    eval_logs_0 = {}
    eval_logs_0["eval_method"] = '1 classname-containing caption'
    eval_logs_0["selection_mask_name"] = ''
    eval_logs_0["top_1_accuracy, single language point"] = [100*caption_accuracy_metric.compute().item() for _ in range(len(cls_weight_acc))]
    eval_logs_0["cls_weight"] = cls_weight_acc
    eval_logs_0["run_params"] = run_params

    save_eval_data('non_ensembled_standard_clip',eval_logs_0,run_params)


def eval_cls_ful_des_plus_cls_less_des_classwise(caption_encodings,description_encodings,selection_mask_acc,class_indices_str,run_params,selection_mask_name,global_image_encodings,global_labels):

    language_encodings = torch.cat([caption_encodings,description_encodings],dim=0)
    global_image_description_similarity_matrix = torch.matmul(global_image_encodings, language_encodings.T)

    cls_weight_acc = []
    accuracy_acc = []
    for cls_weight in tqdm(eval(run_params['cls_weight_range'])):
        description_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=len(class_indices_str)).to(run_params['calculation_device'])
        global_images_classwise_descriptions_acc = []

        for class_idx in class_indices_str:
            caption_mask = torch.zeros(len(class_indices_str),dtype=torch.float16,device=run_params['calculation_device'])
            caption_mask[int(class_idx)] = cls_weight
            if torch.sum(selection_mask_acc[class_idx]) != 0:
                selection_mask = torch.cat([caption_mask,(selection_mask_acc[class_idx]/torch.sum(selection_mask_acc[class_idx])).to(run_params['calculation_device'])],dim=0)
            else:
                selection_mask = torch.cat([caption_mask,selection_mask_acc[class_idx]],dim=0)
            global_image_class_description_similarity_matrix = torch.matmul(global_image_description_similarity_matrix,selection_mask)
            global_images_classwise_descriptions_acc.append(global_image_class_description_similarity_matrix)
        
        global_images_classwise_descriptions_acc = torch.stack(global_images_classwise_descriptions_acc)

        description_predictions = torch.argmax(global_images_classwise_descriptions_acc,dim=0)

        global_labels = global_labels.to(run_params['calculation_device'])

        description_accuracy_metric(description_predictions,global_labels)

        cls_weight_acc.append(float(cls_weight))
        accuracy_acc.append(100*description_accuracy_metric.compute().item())

    eval_logs = {}
    eval_logs["eval_method"] = '1 classname-containing description + classwise, classname-free descriptions, global eval'
    eval_logs["selection_mask_name"] = selection_mask_name
    eval_logs["top_1_accuracy, language ensembling"] = accuracy_acc
    eval_logs["cls_weight"] = cls_weight_acc
    eval_logs["run_params"] = run_params

    save_eval_data(f'classname_free_{selection_mask_name}',eval_logs,run_params)


def eval_cls_ful_descriptions_classwise(cls_description_encodings_dict,global_eval_image_encodings,global_eval_labels,run_params,class_indices_str,selection_mask_name):

    description_ensembling_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=len(class_indices_str)).to(run_params['calculation_device'])
    description_maxing_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=len(class_indices_str)).to(run_params['calculation_device'])

    sim_acc_ensembling = torch.zeros((len(global_eval_image_encodings),len(class_indices_str)),dtype=torch.float16,device=run_params['calculation_device'])
    sim_acc_maxing = torch.zeros((len(global_eval_image_encodings),len(class_indices_str)),dtype=torch.float16,device=run_params['calculation_device'])

    for i,(class_idx_str, description_encodings_tensor) in enumerate(cls_description_encodings_dict.items()):
        ensembled_similarities = torch.einsum('ij,mj->im',global_eval_image_encodings,description_encodings_tensor).mean(dim=1)
        maxed_similarities = torch.einsum('ij,mj->im',global_eval_image_encodings,description_encodings_tensor).max(dim=1).values
        sim_acc_ensembling[:,i] = ensembled_similarities.clone()
        sim_acc_maxing[:,i] = maxed_similarities.clone()
    
    predictions_acc_ensembling = torch.argmax(sim_acc_ensembling,dim=1)
    predictions_acc_maxing = torch.argmax(sim_acc_maxing,dim=1)
        
    description_ensembling_accuracy_metric(predictions_acc_ensembling,global_eval_labels)
    description_maxing_accuracy_metric(predictions_acc_maxing,global_eval_labels)

    eval_logs = {}
    eval_logs["eval_method"] = 'classname-containing descriptions classwise, global eval'
    eval_logs["selection_mask_name"] = selection_mask_name
    eval_logs["cls_weight"] = [float(cls_weight) for cls_weight in eval(run_params['cls_weight_range'])]
    eval_logs["top_1_accuracy, language ensembling"] = [100*description_ensembling_accuracy_metric.compute().item() for _ in range(len(eval_logs["cls_weight"]))]
    eval_logs["top_1_accuracy, language maxing"] = [100*description_maxing_accuracy_metric.compute().item() for _ in range(len(eval_logs["cls_weight"]))]
    eval_logs["run_params"] = run_params

    save_eval_data(f'classname_containing_{selection_mask_name}',eval_logs,run_params)