#  Human Action Recognition Project

##  Overview
This project implements a **human action recognition system** using two different deep learning approaches:
1. A **3D CNN model built from scratch**
2. A **transfer learning model** using MobileNetV2 with LSTM layers

The system classifies human actions into 6 categories: calling, sitting, standing, walking, waving, and writing.

## Key Components

### 1. Data Preparation
- **Dataset**: Contains videos organized in 6 action classes
- **Preprocessing**:
  - Converted MOV files to MP4 format
  - Extracted 30 frames per video (resized to 224x224)
  - Applied class balancing via downsampling
  - Performed data augmentation (flipping, cropping, speed changes)

### 2. Model Architectures
####  3D CNN Model
- 3D convolutional layers with max pooling
- Batch normalization
- Dense layers with dropout
- **Parameters**: ~19.9M trainable

####  MobileNetV2 + LSTM
- MobileNetV2 for spatial feature extraction
- Bidirectional LSTM for temporal processing
- **Parameters**: ~3.7M total (1.5M trainable)

### 3. Training & Evaluation
- **Dataset Split**:
  - Train: ~70%
  - Validation: ~15% 
  - Test: ~15%
- **Performance**:
  - 3D CNN: 70% test accuracy
  - MobileNet: 85% test accuracy

## 📊 Results Summary

| Model          | Test Accuracy | Test Loss | 
|----------------|--------------|-----------|
| 3D CNN         | 70%          | 34.71     | 
| MobileNet+LSTM | 85%          | 0.47      |    

**Confusion matrices** and **training curves** are available in the notebook.

##  How to Use
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook Human_Action_recognition.ipynb`

##  File Structure
```
├── Human_Action_recognition.ipynb  # Main notebook
├── converted_videos/               # MP4 videos
├── processed_frames/               # Extracted frames  
├── downsampled/                    # Balanced dataset
├── augmented/                      # Augmented data
└── models/                         # Saved models
```

##  Key Findings
- Transfer learning approach outperformed the scratch model by 15%
- MobileNet+LSTM showed better generalization (lower test loss)
- 3D CNN suffered from higher variance (large gap between train/val accuracy)

##  Future Improvements
- Try larger pretrained models (EfficientNet, ViT)
- Implement attention mechanisms
- Add more diverse action classes
- Optimize for real-time performance
