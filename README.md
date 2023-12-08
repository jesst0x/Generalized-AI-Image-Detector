# Generalized AI Image Detector
This is the code implementation of the project **Generalized Detector of AI-Generated Images** to develop a discrimator model that could generalize to unseen generator architectures.



## Requirement

Setting up a virtual environment as following is recommended:-

```
conda env create -n detector python=3.10
conda activate detector
pip install -r requirements.txt
```


## Data Preparation
1. You can download real images datasets used in the project, [Celeba](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [FFHQ](https://github.com/NVlabs/ffhq-dataset).
2. For synthetic images dataset, please heads to the official github page of generator model to downlaod code, pretrained model checkpoint and follow the instruction there to do inference to generate the images.\
[Progan](https://github.com/tkarras/progressive_growing_of_gans), [Stylegan](https://github.com/NVlabs/stylegan), [Stylegan2](https://github.com/NVlabs/stylegan2), [Stylegan3](https://github.com/NVlabs/stylegan3), [VQGan](https://github.com/CompVis/taming-transformers#celeba-hq), [StyleSwin](https://github.com/microsoft/StyleSwin), [StyleNat](https://github.com/SHI-Labs/StyleNAT), [ProjectedGan](https://github.com/autonomousvision/projected-gan), [LDM](https://github.com/CompVis/latent-diffusion), [WaveDiff](https://github.com/VinAIResearch/WaveDiff), [RDM](https://github.com/THUDM/RelayDiffusion/tree/main)

3. Resize the images to `256x256` by running 

```
python process_dataset.py --dataset_dir <path/to/dataset_dir> --output_dir <path/to/dir_to_save_images>
```
There are other arguments `image_name` and `count` if you want to prefix resize image file name or limit number of images.

4. Note that the images selected might not be the same as this project, due to random seeds selected and subset of dataset.

## Training

To train the model, run the following:-

```
python generate_evaluate.py --synthetic_train_dir <path/to/synthetic_img_training_set> --real_train_dir <path/to/real_img_traing_set> --synthetic_eval_dir <path/to/synthetic_img_validation_set> --real_eval_dir <path/to/real_img_validation_set> --loggin_dir <path/to/log_model_and_result> --is_resnet <run resnet or adaboost>
```

There are additional hyperparameter arguments

- `batch_size`
- `epoch`: Number of epoch to run algorithm
- `learning_rate`: Initial learning rate of adaptive learning or else fixed
- `decay_rate`: Decay rate of adaptive learning rate (default 1 means not adaptive)
- `decay_epoch`: Number of epoch before next decay (staircase)
- `freeze_layer`: Freeze first n layers of Resnet50
- `nn_layer`: Adaboost base estimators layers list

Ref to the code for other optional arguments. All training model checkpoints are saved in `logging_dir`. This allows us to loads weight to do evaluation or resume training in future. 

## Optimal Threshold
To find optimal threshold of trained model by using roc, run the following command:

```
python threshold.py --synthetic_eval_dir <path/to/synthetic_validation_set> --real_eval_dir <path/to/real_image_validatation_set> --model_dir <path/to/trained_model_dir> --resnet_checkpoint <path/to/resnet_model_checkpoint>
```

You can choose which checkpoint that you have trained by using `resnet_checkpoint` argument.

## Evaluation

To evaluate the performance of trained model on test set, run the following command:-

```
python general_evaluate.py --synthetic_eval_dir <path/to/synthetic_validation_set> --real_eval_dir <path/to/real_image_validatation_set> --model_dir <path/to/trained_model_dir> --resnet_checkpoint <path/to/resnet_model_checkpoint>
```

Optional arguments
- `threshold`: Threshold to classify the images, used to compute threshold dependent metrics (accuracy, precision etc). Default 0.5 if unset
- `save_img`: Whether to save misclassfied images for error analysis.

