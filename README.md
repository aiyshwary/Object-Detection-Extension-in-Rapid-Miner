# 🧾 Object Detection Extension for RapidMiner

This extension enables users to **fine-tune pre-trained object detection models** and perform **inference on new images** within RapidMiner Studio using transfer learning.

Supported models:
- Faster R-CNN  
- FCOS  
- SSD  
- SSDLite  
- RetinaNet  

---

## ⚙️ Operators Included

- **FineTuneObjectDetectionModel**  
  Fine-tunes a pre-trained object detection model using a labeled image dataset.

- **ObjectDetectionModelInference**  
  Performs inference on new images using the fine-tuned model and outputs predictions.

---

## 🛠️ Installation & Setup

### 1. Required RapidMiner Extensions

Install the following from RapidMiner's Marketplace:
- `Custom Operators >= 1.1.0`
- `Operator Toolbox >= 2.17.0`
- `Image Handling >= 0.2.1`
- `Python Scripting >= 10.0.1`

Make sure Python Scripting is properly configured with a valid Python environment.

---

### 2. Python Environment Setup

#### Step 1: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

#### Step 2: Create Environment
```bash
conda create -n rm_obj_detection python=3.10.8 numpy=1.23.2 pandas=1.5.2 -c conda-forge
conda activate rm_obj_detection
Step 3: Install Required Packages
For CPU-only:

bash
Copy
Edit
pip install torch torchvision torchaudio
For GPU (CUDA-enabled systems):

bash
Copy
Edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Additional Packages:

bash
Copy
Edit
pip install pillow matplotlib chardet
3. Add the Extension JAR
Place the .jar file inside the following directory:

makefile
Copy
Edit
C:\Users\<your_username>\.RapidMiner\extensions
Restart RapidMiner Studio after adding the file.

🚀 How to Use
🏋️‍♀️ FineTuneObjectDetectionModel
Inputs:

Images directory

Annotations directory (e.g., YOLO or Pascal VOC format)

Results directory

Optional: tuning hyperparameters

Outputs:

model_state_dict.pth – Fine-tuned model weights

class_list_df.csv – CSV file with detected class names

These are used later for inference.

🔍 ObjectDetectionModelInference
Inputs:

Path to the fine-tuned model (.pth)

Image path to run inference on

Class list CSV from training

Output:

DataFrame with predicted class labels and bounding boxes

Optionally, the image can be saved with overlaid predictions
