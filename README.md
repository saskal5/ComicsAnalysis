# ComicsAnalysis
Automation of comics analysis

![frm_thesis_diagram_v3](https://github.com/saskal5/ComicsAnalysis/assets/43573699/57d60a37-bc93-461f-9ac5-570125234171)

PDFs_HQ file contains every panel image from the entire Tintin comic series. For copyright issues they are not provided here.

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

