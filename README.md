# Layout Element Detection on PDF Documents - SciTech Patent Art

## Overview

This project focuses on detecting **Layout Elements** (such as text, tables, figures, and titles) from PDF documents, aiming to achieve performance comparable to **Azure Layout Parser**. Although the desired output wasn't fully achieved, significant efforts were made, including training custom models on the **YOLOv8** architecture.

## Project Goals

- **Detect Layout Elements from PDF documents** with high accuracy.
- **Automate document structure parsing** to improve efficiency and consistency.
- **Achieve results similar to Azure Layout Parser**, leveraging object detection models.

## Initial Approach

### 1. PubLayNet Dataset Training

To kick off the project, I initially trained the model on a **small subset of the PubLayNet dataset**. The dataset was converted to YOLOv8 format using a Python script I developed, which is available in this repository.

- **Training Setup**: 
  - **Platform**: Google Colab with a T4 GPU.
  - **Epochs**: 500 epochs.
  - **Batch Size**: 16.
  
  The initial results were not as promising as expected, falling short of the performance I had hoped to achieve.

### 2. Manual Annotation of Company Dataset

Realizing the limitations of the initial model, I turned to manual annotation using **Roboflow**. The dataset was built from **7,000 images** specific to **SciTech Patent Art**'s requirements. Here's the process I followed:

- **Manual Annotation**: I started with 500 images, and after training a model with this subset, I used **Roboflow's auto-detection** feature to annotate another set of 500 images. This iterative approach continued until all 7,000 images were annotated.
  
  Special thanks to [@VishalMendhikar](https://github.com/VishalMendhikar), my colleague, who helped significantly with the manual annotation process.

- **Training Setup**:
  - **Pretrained Weights**: The model was fine-tuned using weights trained for 100 epochs on the PubLayNet dataset.
  - **Platform**: Google Colab with an A100 GPU.
  - **Epochs**: 500 epochs.
  - **Batch Size**: 16.
  - **Time Taken**: 7+ hours.

  This model performed exceptionally well on **SciTech Patent Art's dataset**, but it did not generalize well to other datasets, as it lacked diversity (i.e., PubLayNet dataset was not included in the fine-tuning due to resource limitations).

### 3. GPU Resource Limitations

Due to the lack of sufficient **GPU resources**, I couldn’t train the model on the combined **SciTech Patent Art dataset** and the **PubLayNet dataset**. If more GPU resources become available in the future, the model could be trained on the full dataset to improve its generalization capability.

This documentation is intended to store this information and all code for future reference, so when more resources are available, the team can quickly pick up where we left off and proceed with the training.

## Modern Approach: ComfyUI Workflow

As an alternative, I experimented with a modern workflow using **ComfyUI** with the YOLOv8 model for layout detection. While the accuracy was more than **70%**, the workflow was not viable for production due to its heavy GPU requirements and long processing times.

## Pretrained Models: Explorations and Limitations

Throughout the project, I explored several pretrained models for layout element detection. However, none of the models produced results that met the accuracy requirements or matched the expected output.

---

## Code and Resources

1. **Python Script for PubLayNet Conversion to YOLOv8 Format**: [[Link to PubLaynet to yolo v8](https://github.com/Abhik35/layout-element-detection/blob/main/PubLayNet%20to%20YOLOv8%20Format/convert_PubLayNet_model.py)]
2. **Model Training on Google Colab**: [[Link to Colab notebook](https://github.com/Abhik35/layout-element-detection/blob/main/Model%20Training/train_yolov8_object_detection_on_custom_dataset.ipynb)]
3. **Model Weights**: [[Links to weights and further training details](https://github.com/Abhik35/layout-element-detection/tree/main/Model%20Weights)]
4. **Roboflow Integration**: [[Roboflow project link](https://universe.roboflow.com/patent-jskng/table-figure-detection/dataset/22)]
5. **Comfyui workflow**: [[Comfyui workflow link](https://github.com/prodogape/Comfyui-Yolov8-JSON)]

---

## Future Directions

- **Expand Training**: Once additional GPU resources are available, the plan is to retrain the model with both the **SciTech Patent Art dataset** and the **PubLayNet dataset** for better generalization.
- **Explore New Architectures**: Investigate more efficient architectures that can deliver real-time performance, even with limited GPU availability.
- **Optimize ComfyUI Workflow**: Further optimize the ComfyUI pipeline for layout detection.

---

### Acknowledgments

- Thanks to [@VishalMendhikar](https://github.com/VishalMendhikar) for his assistance with manual annotation.
- Special recognition to the **Roboflow platform** for easing the annotation and detection workflow.

---

This documentation serves as a roadmap for future improvements, aiming to help the team at **SciTech Patent Art** refine and expand the current model for layout element detection.
