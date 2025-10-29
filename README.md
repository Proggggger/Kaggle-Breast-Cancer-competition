# Kaggle-Breast-Cancer-competition
# Breast Cancer Detection Challenge Solutions

🚀 **Try the deployed application:** [Breast Cancer Detection App](https://breast-cancer-detection-app.streamlit.app/)

This repository contains the code and resources for my solutions to the Kaggle competition: [Breast Cancer Detection](https://www.kaggle.com/competitions/breast-cancer-detection/overview).

The goal of this competition is to build a machine learning model capable of accurately classifying breast cancer as either benign or malignant based on various features extracted from breast mass images.

## Repository Contents

This repository is organized to provide a clear and understandable structure for the project. You will find the following:

* **`notebooks/`**: This directory contains Jupyter Notebooks detailing the different stages of the project, including:   
    * **`Simple AdaBoost Solution`**: This notebook implements a basic AdaBoost solution for the Breast Cancer detection task. It utilizes key libraries such as:
        * `pandas` for data manipulation and analysis
        * `numpy` for numerical computations
        * `plotly.express` and `plotly.graph_objects` for data visualization


* **`data/`**: This directory will likely store the competition data (train, test) and potentially any intermediate data files generated during the data processing or feature engineering stages.



* **`requirements.txt`**: This file lists the Python libraries and their versions used in this project. You can use `pip install -r requirements.txt` to install all the necessary dependencies.

* **`README.md`**: This file (the one you are currently reading) provides an overview of the repository and the project.

## Approach and Key Findings

This repository showcases my approach to tackling the breast cancer detection challenge. Key aspects of the solution might include:

* **Careful Data Preprocessing**: Handling missing values, dealing with outliers, and scaling/normalizing features to ensure optimal model performance.
* **Strategic Feature Selection/Engineering**: Identifying the most relevant features and potentially creating new informative features from the existing ones.
* **Rigorous Model Selection**: Experimenting with different classification algorithms and tuning their hyperparameters using techniques like GridSearchCV or RandomizedSearchCV to find the best performing model.
* **Robust Evaluation**: Employing cross-validation strategies to obtain reliable estimates of the model's generalization ability.
* **Ensemble Methods**: Exploring the potential benefits of combining multiple models to improve prediction accuracy and robustness.

Further details about the specific models used, feature engineering techniques, and evaluation results can be found in the individual notebooks within the `notebooks/` directory.

## How to Use This Repository

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd breast-cancer-detection
    ```

2.  **Download the competition data:** Navigate to the [Kaggle competition page](https://www.kaggle.com/competitions/breast-cancer-detection/overview) and download the `train.csv` and `test.csv` files. Place these files in the `data/` directory.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Explore the notebooks:** Open and run the Jupyter Notebooks in the `notebooks/` directory in sequential order to understand the data exploration, feature engineering, model training, and evaluation processes.

5.  **Reproduce the submission:** Every solution notebook generates it's own `submission.csv` file which can be uploaded to kaggel.

## Contributions and Improvements

Feel free to explore the code and suggest improvements or report any issues. Contributions are welcome!

## Deployed Application

This repository includes a deployed Streamlit web application that provides an interactive interface for breast cancer detection using the trained AdaBoost model. The application offers the following features:

### Key Features:
- **Data Exploration**: Interactive visualization of the breast cancer dataset with 3D scatter plots
- **Model Performance**: View confusion matrix and calibration curve to understand model behavior
- **Manual Prediction**: Input feature values to get real-time predictions from the trained model
- **Visual Feedback**: Probability gauge and 3D visualization showing where your input falls relative to training data

### How to Use:
1. Visit the [Breast Cancer Detection App](https://breast-cancer-detection-app.streamlit.app/)
2. Explore the dataset visualization to understand feature distributions
3. Review model performance metrics
4. Enter feature values in the manual prediction section to get a diagnosis prediction
5. View the 3D visualization to see how your input compares to the training data

The application uses a pre-trained AdaBoost classifier that analyzes 30 different features extracted from breast mass images to classify tumors as either benign or malignant. The model achieves high accuracy on the training data and provides probability estimates for each prediction.

## Disclaimer

The solutions provided in this repository were developed for a specific Kaggle competition and may not be directly applicable to other datasets or problems without modification. The results achieved in the competition are specific to the given dataset and evaluation metrics.