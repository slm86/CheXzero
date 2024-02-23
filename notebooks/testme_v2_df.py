import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append('/p/project/ccstdl/moroianu1/chexzero_main/CheXzero/')

from eval import evaluate, bootstrap
from zero_shot import make_true_labels, run_softmax_eval, CXRTestDataset

import open_clip
import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

## Define Zero Shot Labels and Templates

# ----- DIRECTORIES ------ #
cxr_filepath: str = '../data/chexpert_test.h5' # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[str] = '../data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
model_dir: str = '../checkpoints/chexzero_weights' # where pretrained models are saved (.pt) 
predictions_dir: Path = Path('../predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

context_length: int = 77

pd.set_option('display.max_columns', None)

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

# ----- MODEL PATHS ------ #
# If using ensemble, collect all model paths
model_paths = []
for subdir, dirs, files in os.walk(model_dir):
    for file in files:
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)
# TODO: manually place here model path you want e.g. model_paths = ['path1.pt']    
checkpoint_folder = '/p/project/ccstdl/moroianu1/open_clip_main/logs/2024_02_20-02_11_31-model_ViT-B-32-lr_0.001-b_512-j_4-p_amp/checkpoints/'

def make(
    model_path: str, 
    cxr_filepath: str, 
    pretrained: bool = True, 
    context_length: bool = 77, 
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels. 
    
    args: 
        * model_path - String for directory to the weights of the trained clip model. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.
    
    Returns model, data loader. 
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=model_path)
    model = model.to(device) 
    # load data
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    # if using CLIP pretrained model
    if pretrained: 
        # resize to input resolution of pretrained clip model
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    
    # create dataset
    torch_dset = CXRTestDataset(
        img_path=cxr_filepath,
        transform=transform, 
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)
    
    return model, loader

## Run the model on the data set using ensembled models
def ensemble_models(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths) # ensure consistency of 
    for path in model_paths: # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path, 
            cxr_filepath=cxr_filepath, 
        ) 
        
        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None: 
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else: 
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction
        if cache_dir is not None and os.path.exists(cache_path): 
            print("Loading cached prediction for {}".format(model_name))
            y_pred = np.load(cache_path)
        else: # cached prediction not found, compute preds
            print("Inferring model {}".format(path))
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None: 
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg

results_df = pd.DataFrame()
print('Number epochs:',len(os.listdir(checkpoint_folder))-1)
n_epochs = len(os.listdir(checkpoint_folder))-1
for ep in range(1,1+n_epochs):
    checkpoint = os.path.join(checkpoint_folder, 'epoch_'+str(ep)+'.pt')
    model_paths = [checkpoint]

    predictions, y_pred_avg = ensemble_models(
        model_paths=model_paths, 
        cxr_filepath=cxr_filepath, 
        cxr_labels=cxr_labels, 
        cxr_pair_template=cxr_pair_template, 
        cache_dir=cache_dir,
        save_name='mimic-epoch-'+str(ep)
    )

    # # save averaged preds
    # pred_name = "chexpert_preds.npy" # add name of preds
    # predictions_dir = predictions_dir / pred_name
    # np.save(file=predictions_dir, arr=y_pred_avg)

    # make test_true
    test_pred = y_pred_avg
    test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

    # evaluate model
    cxr_results = evaluate(test_pred, test_true, cxr_labels)

    cxr_results['epoch'] = ep

    results_df = pd.concat([results_df,cxr_results], ignore_index=True)
    # # boostrap evaluations for 95% confidence intervals
    # bootstrap_results = bootstrap(test_pred, test_true, cxr_labels)

    # print(bootstrap_results[1])
print(results_df)
results_df.to_csv('testme_v2_plot.csv')