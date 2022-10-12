# AIDGN
PyTorch Implementation for "Domain Generalization Through the Lens of Angular Invariance" (Our code is mainly based on [DomainBed, Ishaan and David, 2021](https://github.com/facebookresearch/DomainBed) library.)

## Usage

1. Obtain dataset
The datasets used in our experiments can be downloaded from the following links:
* PACS: https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd
* VLCS: https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8  
* OfficeHome: https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC
* TerraIncognita: https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz
                  https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip



2. Train the model
```
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/OfficeHome/\
       --algorithm AIDGN\
       --dataset OfficeHome\
       --test_env 2
```
or launch a sweep
```
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher\
       --algorithms AIDGN\
       --datasets OfficeHome\
       --n_hparams 20\
       --n_trials 3 
```
3. To view the results of your sweep:

````sh
python -m domainbed.scripts.collect_results\
       --input_dir=/my/sweep/output/path
````

## Requirements
* Python 3.7.9
* PyTorch 1.7.1
* torchvision 0.8.2
* Numpy 1.19.4


