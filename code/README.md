# Code

ST-VQA model
-----

![](../resources/tgif_model.png "TGIF-QA")



Setup Instructions
-----

1. Install python modules and TensorFlow v0.10.0

    ```bash
    pip install -r requirements.txt
    python -m spacy.en.download
    # TF for GPU (you need CUDA 7.5 and cuDNN 6.0):
    pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
    # TF for CPU:
    #pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
    ```

2. Set TGIF-QA dataset and related files in code folder.

    ```bash
    # in code folder
    mkdir dataset
    mkdir dataset/tgif
    cp -r ../dataset dataset/tgif/DataFrame
    mkdir dataset/tgif/features dataset/tgif/Vocabulary
    ```

3. Download GIF files in [dataset](../dataset/README.md) page and extract the zip file it into `dataset/tgif/gifs`.



Pre-processing the visual features
-----

1. Download GIF files into your directory.

2. Install ffmpeg.

3. Extract all GIF frames into a separate folder:

    ```bash
    ./save-frames.sh dataset/tgif/{gifs,frames}
    ```

4. Extract [ResNet-152](https://github.com/KaimingHe/deep-residual-networks) and [C3D](https://github.com/facebook/C3D) features by using each pretrained models.
    - Extract 'res5c', 'pool5' for ResNet-152, and 'conv5b', 'fc6' for C3D.
    - If a GIF file contains less than 16 frames, append the last frame to have 16 frames at least.
    - When extracting the C3D features, use stride 1 pad the first frame eight times for the first frame, and pad the last frame 7 time for the very last frame (SAME padding).

5. Wrap each extracted features into hdf5 files per layer, name them as 'TGIF_[MODEL]_[layer_name].hdf5' (ex, TGIF_C3D_fc6.hdf5, TGIF_RESNET_pool5.hdf5), and save them into 'code/dataset/tgif/features'. For example, pool5 feature and res5c feature need to be stored in a different hdf5 file. Each feature file should have to be a dictionary that uses 'key' field of each dataset file as the key of a dictionary and a numpy array of extracted features in (\#frames, feature dimension) shape. 



Note. We uploaded the two hdf5 files (  [Resnet_pool5](https://drive.google.com/file/d/0B15H16jpV4w2SlVleTBRT3dUTGs/view?usp=sharing), [C3D_fc6](https://drive.google.com/file/d/0B15H16jpV4w2cFZoOXpPMlFLX3M/view?usp=sharing) ), but we failed to upload the other two files because of its size.





Training
-----

* Choose task [Count, Action, FrameQA, Trans] and model name [C3D, Resnet, Concat, Tp, Sp, SpTp]
* Run python script
    ```
    cd gifqa
    python main.py --task=Count --name=Tp
    ```



Evaluation
-----

* Choose task [Count, Action, FrameQA, Trans], model name [C3D, Resnet, Concat, Tp, Sp, SpTp] and set checkpoint path
* Run python script
    ```
    cd gifqa
    python main.py --task=Count --name=Tp --checkpoint_path=YOUR_CHECKPOINT_PATH --test_phase=True --evaluate_only=True
    ```



Run Pretrained Models
-----

* Download checkpoints for concat and temporal models from [this link](https://drive.google.com/file/d/1EwbGkGviK6FOOI8UXQu0vdb4ELFBSoBU/view?usp=sharing) and place checkpoint folders in `gifqa/pretrained_models`
* Run test script
    ```
    cd gifqa
    ./test_scripts/{task}_{model}.sh
    ```



## Notes

Last Edit: December 02, 2017
