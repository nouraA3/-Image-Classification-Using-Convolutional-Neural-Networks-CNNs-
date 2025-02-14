# -Image-Classification-Using-Convolutional-Neural-Networks-CNNs-
 This project focuses on image classification using Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset. The model is optimized to achieve higher accuracy using data augmentation, dropout regularization, batch normalization, and advanced optimizers

📊 Overview

The CIFAR-10 dataset contains 60,000 color images, categorized into 10 different classes. The goal is to enhance classification accuracy by implementing an optimized CNN architecture.

📂 Dataset and Preprocessing

✅ Dataset: CIFAR-10 (60,000 images, 10 classes).
✅ Preprocessing Steps:
	•	Split into training & testing sets.
	•	Normalize pixel values to [0,1] for faster convergence.
	•	Apply data augmentation to improve generalization:
	•	Rotation: 15 degrees
	•	Width shift: 10%
	•	Height shift: 10%
	•	Horizontal flipping: Enabled
	•	Zooming: 10%

 🛠 CNN Model Architecture

To improve classification performance, the CNN model consists of:

✅ Convolutional Layers:
	•	6 convolutional layers with increasing filter sizes (32, 64, 128).
	•	ReLU activation function for non-linearity.

✅ Batch Normalization:
	•	Applied after each convolutional layer to stabilize learning.

✅ Pooling Layers:
	•	MaxPooling layers reduce dimensionality after every two convolutional layers.

✅ Dropout Regularization:
	•	Applied at different levels: (0.3, 0.4, 0.5) to prevent overfitting.

✅ Fully Connected (Dense) Layers:
	•	A Dense layer with 128 neurons followed by Dropout.
	•	A final output layer with 10 neurons for classification.

 🚀 Model Compilation and Training

✅ Baseline Model Performance:
	•	Test Accuracy: 72.06%
	•	Loss: 1.2132

✅ Optimized Model (After Improvements):
	•	Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
	•	Loss Function: Sparse Categorical Crossentropy
	•	Learning Rate Scheduling: Reduce LR by 5% every 10 epochs
	•	Early Stopping: Stop training if validation accuracy does not improve for 10 epochs.
	•	Epochs: 50
	•	Batch Size: 64

📈 Final Model Performance:
	•	Test Accuracy: 85.30%
	•	Loss: 0.4409
	•	Prediction Accuracy: 95.46%

 📢 Future Enhancements

To further improve performance, we recommend:
	•	Exploring deeper architectures like ResNet or EfficientNet.
	•	Fine-tuning hyperparameters using advanced search techniques.
	•	Expanding data augmentation strategies for better generalization.
	•	Using optimizers like Stochastic Weight Averaging (SWA) for stability.
