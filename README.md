# Document Image Classification

## Project Setup Steps
1. Clone this repository
2. Create docker container
    ```
    docker build -t doc-image-classification .
    ```
3. To prepare dataset and run training pipeline, enter interactive mode
    Dataset should be present in working directory (./dataset/original)
    ```
    # mount dataset and start interactive shell
    docker run -it -v ./dataset:/doc-image-classification/dataset doc-image-classification sh 
    ```
    Now in interactive shell
    ```
    # prepare dataset
    python app/prepare_dataset.py

    #train model
    # use python app/main.py for help on training arguments
    # below is an example command
    python app/main.py --epochs 50 --exp "resnet-imagenet-all" --lr 0.001 --model_name resnet --save_path renet_all.pth
    ```

    To run stramlit application
    ```
    docker run -p 8000:8000 doc-image-classification 
    ```
*Note that the model name in streamlit app should match with model name present in working direcotry*
*You need to install nvidia-container-toolkit to use cuda in docker container*
