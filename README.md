# ComicsAnalysis

![comics_reading_robot](https://github.com/saskal5/ComicsAnalysis/assets/43573699/9b7643dc-5203-4ea4-9aef-14aa79d862e4)

Comics analysis has grown in popularity, but labeling comics is still time-consuming and costly, leading to limited datasets. Deep learning models have been used to detect objects in comics, however they struggle to identify main characters, causing information loss and inadequate large-scale analysis. This study introduces a new framework to automate object identification and large-scale analysis in comic series. The framework involves extensive data collection, identification of panels, characters, speech balloons, and text, and automation of sequential information extraction. Applied to the Tintin series, it uses edge detection and the YOLOv8 model for high-performance character and speech balloon detection, and TrOCR for text recognition, significantly reducing manual labeling efforts.


Model files, notebooks, and evaluation outputs of the techniques applied can be found in this Drive ![folder](https://drive.google.com/drive/folders/1JnlWMVtd7Gb-z3Un-SqK7mF7Ewy9MMMH)

## Table of contents
* [Methodology](#mmethodology)
* [Datasets](#datasets)
* [Model Performances](#model-performances)

## Methodology

![frm_thesis_diagram_v3](https://github.com/saskal5/ComicsAnalysis/assets/43573699/57d60a37-bc93-461f-9ac5-570125234171)

Naturally, the first step is the collection of the comic book PDFs which follows panel extraction in which the panel images from each book are stored separately. After this step, the main character, speech balloon and expression datasets are created. With these datasets, YOLOv8 models are trained and these models are used in various operations. The results are evaluated with different techniques and in the final stage of the research, certain analyses are performed.

## Datasets

In the context of this study, three distinct datasets were prepared. First, a dataset targeting main character detection was constructed. This dataset comprised 32 panels randomly selected from each of the 23 books, ensuring coverage of at least one main character per panel alongside two background panels without any main character. Next, a separate dataset was created specifically for speech balloon detection. In this dataset, 20 random panels featuring at least one speech balloon alongside one random panel without any speech balloon were selected from each book. Finally, a third dataset was devised to target the detection of single exclamation or question marks (onomatopoeia) frequently encountered in the Tintin series, which were mostly undetectable by the text recognition model used. 

All datasets in this study were created on Roboflow platform.

|       Dataset      |                                                        See It In Roboflow                                                          | 
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | 
|   Main Characters  |                                   [URL](https://universe.roboflow.com/azat/tintin_hq/dataset/3)                                    |
|   Speech Balloons  |                                   [URL](https://universe.roboflow.com/azat/speech_balloons_hq/3)                                   |
|     Expression     | [URL1](https://universe.roboflow.com/azat/exclamation-question/1) + [URL2](https://universe.roboflow.com/azat/exclamation-question/3) |




## Model Performances

<details>
<summary>Main Character Detection</summary>



|     Model     |     Epoch     |   Time   |    P    |   R   |  mAP50  | mAP50-95 |
| ------------- | ------------- | -------- | ------- | ----- | ------- | -------- |
|    YOLOv5     |     50        |  12.47h  |  0.934  | 0.911 |  0.936  |  0.844   |
|    YOLOv8     |     62        |  7.11h   |  0.851  | 0.838 |  0.885  |  0.737   |
|    YOLOv9     |     50        |  1.69h   |  0.975  | 0.955 |  0.984  |  0.956   |


</details>

<details>
<summary>Speech Balloon Detection</summary>


|     Model     |     Epoch     |   Time   |    P    |   R   |  mAP50  | mAP50-95 |
| ------------- | ------------- | -------- | ------- | ----- | ------- | -------- |
|    YOLOv5     |     25        |  3.89h   |  0.974  | 0.987 |  0.991  |  0.888   |
|    YOLOv8     |     25        |  2.72h   |  0.867  | 0.982 |  0.991  |  0.847   |
|    YOLOv9     |     20        |  0.48h   |  0.993  | 0.978 |  0.992  |  0.977   |

</details>

<details>
<summary>Expression Detection</summary>

The expression dataset was only trained with YOLOv8.


|   **Model**      |  **All**    | **Question** | **Exclamation** |   
| ---------------- | ----------- | ------------ | --------------- | 
|   **Precision**  |    0.958    |     0.922    |      0.995      | 
|   **Recall**     |    0.917    |     0.833    |      1.000      | 
|   **mAP50**      |    0.978    |     0.962    |      0.995      | 
|   **mAP50-95**   |    0.629    |     0.624    |      0.634      |

</details>

Some notes regarding the Python codes
* PDFs_HQ file contains every panel image from the entire Tintin comic series. For copyright issues they are not provided here.

