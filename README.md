# VideoQA
This is the implementation of our paper "Video Question Answering via Gradually Refined Attention over Appearance and Motion". 

## Datasets
For our experiments, we create two VideoQA datasets named [**MSVD-QA**](https://mega.nz/#!QmxFwBTK!Cs7cByu_Qo42XJOsv0DjiEDMiEm8m69h60caDYnT_PQ) and [**MSRVTT-QA**](https://mega.nz/#!UnRnyb7A!es4XmqsLxl-B7MP0KAat9VibkH7J_qpKj9NcxLh8aHg). Both datasets are based on existing video description datasets. The QA pairs are generated from descriptions using this [tool](http://www.cs.cmu.edu/~ark/mheilman/questions) with additional processing steps. The corresponding videos can be found in base datasets which are [MSVD](http://www.cs.utexas.edu/users/ml/clamp/videoDescription) and [MSR-VTT](http://ms-multimedia-challenge.com/2016/dataset). For MSVD-QA, [youtube_mapping.txt](https://mega.nz/#!QrowUADZ!oFfW_M5wAFsfuFDEJAIa2BeFVHYO0vxit3CMkHFOSfw) may be needed to build the mapping of video names. The followings are some examples from the datasets.

![MSVD-QA](https://i.imgur.com/KtoT1BZ.png)
![MSRVTT-QA](https://i.imgur.com/gf1ayne.png)

## Models
We propose a model with gradually refined attention over appearance and motion in the video to tackle the VideoQA task. The architecture is presented below. Besides, we also compare the proposed model with three baseline models. Details can be found in the paper.
![model](https://i.imgur.com/RGpZw6V.png)

## Code
The code is written in pure python. [Tensorflow](https://www.tensorflow.org) is chosen to be the deep learning library here. The code uses two implementations of feature extraction networks which are [VGG16](https://github.com/machrisaa/tensorflow-vgg) and [C3D](https://github.com/hx173149/C3D-tensorflow) from the community.

### Environments
* Ubuntu 14.04
* Python 3.6.0
* Tensorflow 1.3.0

### Prerequisits
1. Clone the repository to your local machine.

    ```
    $ git clone https://github.com/xudejing/VideoQA.git
    ```
2. Download the [VGG16 checkpoint](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) and [C3D checkpoint](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0) provided in corresponding repositories, then put them in directory `util`; Download the word embeddings trained over 6B tokens (glove.6B.zip) from [GloVe](https://nlp.stanford.edu/projects/glove/) and put the 300d file in directory `util`.

3. Install the python dependency packages.
    
    ```
    $ pip install -r requirements.txt
    ```

### Usage
The directory `model` contains definition of four models. `config.py` is the place to define the parameters of models and training process.

1. Preprocess the VideoQA datasets, for example:

    ```
    $ python preprocess_msvdqa.py {dataset location}
    ```
2. Train, validate and test the models, for example:

    ```
    $ python run_gra.py --mode train --gpu 0 --log log/evqa --dataset msvd_qa --config 0
    ```
    (Note: you can pass `-h` to get help.)
3. Visualize the training process using tensorboard, for example:

    ```
    $ tensorboard --logdir log --port 8888
    ```

### Citation
If you find this code useful, please cite the following paper:
```
@inproceedings{xu2017video,
  title={Video Question Answering via Gradually Refined Attention over Appearance and Motion},
  author={Xu, Dejing and Zhao, Zhou and Xiao, Jun and Wu, Fei and Zhang, Hanwang and He, Xiangnan and Zhuang, Yueting},
  booktitle={ACM Multimedia}
  year={2017}
}
```
