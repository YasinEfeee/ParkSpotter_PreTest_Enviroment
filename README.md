
# ParkSpotter Project - Test Enviroment

## Overview
The **ParkSpotter** project is designed to detect the occupancy status of parking spots in a simulation environment. Using a toy model, a camera system, and a machine learning model, this system identifies whether a parking space is **EMPTY** or **NOT EMPTY** in real-time.

## Research Methodology
This test project, developed for the ParkSpotter project, focuses on detecting parking space occupancy using machine learning. A physical mockup was created to simulate parking spaces, and data was collected by capturing images with a camera. These images were processed, labeled, and used to train a machine learning model for real-time occupancy detection, with results visualized on-screen.

## Dataset Preparation
To train the model, we created a custom dataset consisting of images taken from a camera placed 66 cm above the toy model. 
The camera captured images that covered the entire parking area.
We cropped images containing parking spaces into smaller sections focusing on the parking areas. 
This was done by selecting specific coordinates in the images and saving the cropped parts for further training, ensuring they were labeled as either "EMPTY" or "NOT EMPTY."
The images were then cropped using Python code and resized to 15x15 pixels for model training.

### Tools and Libraries Used:
- **Python**: Programming language used for model development.
- **PyCharm**: IDE used for development.
- **Scikit-learn**: Machine learning library used for training the SVM model.
- **NumPy**: Used for scientific computations.
- **Pickle**: Used for saving and loading the trained model.
- **OpenCV**: Image processing library used for working with camera images and visualizing results.
- **scikit-image**: Used for image processing tasks like resizing and handling image components.

## Model Training
The model was trained using a Support Vector Machine (SVM) algorithm to classify images as either **EMPTY** or **NOT EMPTY**. We performed hyperparameter tuning using **GridSearchCV** with parameters for **gamma** and **C** to achieve optimal model performance. The model was then evaluated using a test set, and the accuracy was calculated using the **accuracy_score** method.

### Key Steps:
1. **Image Preprocessing**: Each image was resized to 15x15 pixels and flattened into a 1D array.
2. **Model Training**: The SVM model was trained with the dataset, using hyperparameter optimization.
3. **Model Saving**: The trained model was saved for later use using the **Pickle** library.

## Real-Time Parking Spot Detection
After training the model, we developed a main script to use the trained model for real-time parking spot detection. The system processes camera feed images and determines whether each parking spot is **EMPTY** or **NOT EMPTY**.

### System Features:
- **Camera Integration**: Captures live footage of the parking area.
- **Masking**: Converts the captured frame into a mask that identifies parking spot locations.
- **Real-time Processing**: Continuously analyzes the camera feed and visualizes the occupancy status by coloring parking spots green (EMPTY) or red (NOT EMPTY).
- **Connected Components**: Used to identify the regions of interest (parking spots) within the mask.

## File Structure:
```
/project-folder
│
├── /y-clf-data          # Dataset folder with parking images (Empty or Not Empty)
│
├── /y-data              # Parking lot asset mask .png file
│
├── /y-model.p           # Pre-Trained model file (.p)
│
├── /y-model-main.py     # Python scripts for data preprocessing and model training
│
├── /y_main.py           # Python scripts for main operations and real-time detection
│
├── /utils               # Utility scripts for image processing and system integration
│
├── README.md            # Project overview 
│
├── requirements.txt     # Required libraries
│
├── .gitignore           # Git ignore file to exclude unnecessary files
│
└── /Real-life use case # Images of real-life use case
```

## Real-life Use Case
![image alt](https://github.com/YasinEfeee/ParkSpotter_PreTest_Enviroment/blob/acf2c54068d9aed24bbe0c322e40e0499544ca7f/Real-life%20use%20cases/Runing%20example.JPEG)


## Contribute and Support

We're open to any suggestions or ideas to make the project and ourselves better! You can contribute to the project in the following ways:

- **Report Issues and Suggestions:** If you encounter any problems or have improvement ideas, please open an "issue." Every piece of feedback is invaluable to us!
- **Spread the Word:** Share the project with your friends and anyone who might be interested, helping us reach a broader audience.

### Contact

Feel free to reach out to us for more information or to share your contributions. Thank you in advance for your support!
