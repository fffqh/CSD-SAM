# CSD-SAM code hub
![image](assert/output_csd.gif)

## Environments

- torch==1.8.1
- timm==0.4.12
- opencv-python==4.8.0.76
- sscipy==1.10.1
- yacs==0.1.8
- loguru==0.7.2
- pandas==2.0.3
- hdf5storage==0.1.19
- matplotlib==3.4.3
- numpy==1.21.2

## Install SAM

```
cd S3_SAM_CSD
pip install -e .
```

## Prepare for data

- downloaod ordinary train video
  - Baidu link : https://pan.baidu.com/s/1PrRw1Pjmh-Iu8pssVuJW3Q?pwd=12rl
    - pwd：12rl
- put the video in directory `[project]/data`
  - The directory structure is as follows:

    ```
    [project]
    ├─ data
    │    └─ csd.mp4
    ├─ S0_OpticalFlow
    ├─ S1_Classification
    ├─ S2_BuildPrompts
    ├─ S3_SAM_CSD
    ├─ img2video.py
    ├─ requirements.txt
    ├─ run.sh
    ├─ README.md
    └─ video2img.py
    ```

## Run

### Step 1

```
python video2img.py
```

### Step 2

```
./run.sh
```

## Acknowledgments
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)
- [SAM](https://github.com/facebookresearch/segment-anything)

## License
The model is licensed under the [Apache 2.0 license](LICENSE).

