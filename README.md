# Simulated Two-Stream Network with Efficient Distillation for Action Recognition

We introduced a Simulated Two Stream Network (STS-Net), utilizing a more efficient knowledge distillation approach to acquire motion representations. 
First, we try to distill knowledge of optical flow across various levels through a review mechanism, thereby capturing both low-level feature and high-level semantic information.
Second, we apply a decoupled knowledge distillation loss to obtain a more comprehensive knowledge transfer. 
Additionally, we analyzed the role of the activation function in fusing the two streams, and proposed an effective fusion strategy named ``ActivNo".
The experimental results on benchmark datasets (\emph{i.e.}, HMDB51, UCF101, and Kinetics400) demonstrated that the proposed STS-Net achieves superior performance, surpassing comparable methods in terms of efficiency and accuracy.


## Contents
1. [Requirements](#requirements)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Testing](#testing)

## Requirements

Environments:

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```


* Directory tree
 ```
    dataset/
        HMDB51/ 
            ../(dirs of class names)
                ../(dirs of video names)
        HMDB51_labels/
    results/
        test.txt
    trained_models/
        HMDB51/
            ../(.pth files)
```


## Datasets

* The datsets and splits can be downloaded from 

    [Kinetics400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)

    [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

    [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

    [SomethingSomethingv1](https://20bn.com/datasets/something-something/v1)

* To extract only frames from videos 
```
python utils1/extract_frames.py path_to_video_files path_to_extracted_frames start_class end_class
```

* To extract optical flows + frames from videos 
    * Build
    ```
    export OPENCV=path_where_opencv_is_installed

    g++ -std=c++11 tvl1_videoframes.cpp -o tvl1_videoframes -I${OPENCV}include/opencv4/ -L${OPENCV}lib64 -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudaarithm
    
    python utils1/extract_frames_flows.py path_to_video_files path_to_extracted_flows_frames start_class end_class gpu_id
    ```
## Models

Trained models can be found [here](https://drive.google.com/drive/folders/1OVhBnZ_FmqMSj6gw9yyrxJJR8yRINb_G?usp=sharing). The names of the models are in the form of 

```
stream_dataset_frames.pth     

RGB_Kinetics_16f.pth indicates --modality RGB --dataset Kinetics --sample_duration 16
```

For HMDB51 and UCF101, we have only provided trained models for the first split.

## Testing script
For RGB stream:
```
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

For Flow stream:
```
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality Flow --sample_duration 16 --split 1  \
--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

For single stream SFS: 

```
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "trained_models/HMDB51/SFS_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

For two streams RGB+SFS (STS-Net):
```
python test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
--resume_path2 "trained_models/HMDB51/SFS_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

For two streams RGB+Flow:
```
python test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB_Flow --sample_duration 16 --split 1 \
--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
--resume_path2 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51/HMDB51_frames/" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

## Training script
### For RGB stream: 
#### From scratch:
```
 python train.py --dataset Kinetics --modality RGB --only_RGB \
--n_classes 400 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101  \
--frame_dir "dataset/Kinetics" \
--annotation_path "dataset/Kinetics_labels" \
--result_path "results/"
```

#### From pretrained Kinetics400:
```
 python train.py --dataset HMDB51 --modality RGB --split 1 --only_RGB \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/RGB_Kinetics_16f.pth" \
--result_path "results/"
```

#### From checkpoint:
```
 python train.py --dataset HMDB51 --modality RGB --split 1 --only_RGB \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/RGB_Kinetics_16f.pth" \
--resume_path1 "results/HMDB51/PreKin_HMDB51_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx4_varLR2.pth" \
--result_path "results/"
```

### For Flow stream 
#### From scratch:
```
 python train.py --dataset Kinetics --modality Flow \
--n_classes 400 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101  \
--frame_dir "dataset/Kinetics" \
--annotation_path "dataset/Kinetics_labels" \
--result_path "results/"
```

#### From pretrained Kinetics400:
```
 python train.py --dataset HMDB51 --modality Flow --split 1 \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
--result_path "results/"
```

#### From checkpoint:
```
 python train.py --dataset HMDB51 --modality Flow --split 1 \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
--resume_path1 "results/HMDB51/PreKin_HMDB51_1_Flow_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx4_varLR2.pth" \
--result_path "results/"
```

### For SFS:
#### From scratch:  
```
python STS_train.py --dataset Kinetics --modality RGB_Flow \
--n_classes 400 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 \
--output_layers 'avgpool' --SFS_alpha 50 \
--frame_dir "dataset/Kinetics" \
--annotation_path "dataset/Kinetics_labels" \
--resume_path1 "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
--result_path "results/" --checkpoint 1
```

#### From pretrained Kinetics400:
```
python STS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--output_layers 'avgpool' --SFS_alpha 50 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/SFS_Kinetics_16f.pth" \
--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--result_path "results/" --checkpoint 1
```
#### From checkpoint:
```
python STS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--output_layers 'avgpool' --SFS_alpha 50 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/SFS_Kinetics_16f.pth" \
--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--SFS_resume_path "results/HMDB51/SFS_HMDB51_1_train_batch16_sample112_clip16_lr0.001_nesterovFalse_manualseed1_modelresnext101_ftbeginidx4_layeravgpool_alpha50.0_1.pth" \
--result_path "results/" --checkpoint 1
```