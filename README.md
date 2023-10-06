# Fake News Detector

This project is a web-based application for detecting fake news. It leverages machine learning models to classify news articles as real or fake. The main components of the project include data loading, data preprocessing, model training, testing, and prediction.

## Project Structure

The project is organized into several modules:

- **Data Loader (`src/Data/DataLoader.py`)**: This module handles loading the dataset from CSV files containing true and fake news data.

- **Data Preprocessor (`src/data/DataProcess.py`)**: This module preprocesses the loaded data by cleaning and preparing it for training.

- **Model Training (`src/model/model.py`)**: This module defines and trains machine learning models (Logistic Regression, Decision Tree, Random Forest) for classifying news articles.

- **Plot Renderer (`src/plots/render.py`)**: This module provides functions for rendering plots to visualize features.

- **Main Application (`src/main.py`)**: This script orchestrates the entire pipeline by loading data, preprocessing it, training models, and generating visualizations.

- **Streamlit Application (`app.py`)**: This script sets up a web interface using Streamlit, allowing users to enter text for classification and see the predicted result.

- **Prediction Module (`predictor.py`)**: This module contains a function to load the trained model and make predictions on new data.

## Instructions for Running the Project Locally

To run the project locally, follow these steps:

1. Install the necessary dependencies using the provided `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

2. Run the project using Streamlit:
   ```
   streamlit run app.py
   ```

   Alternatively, you can run the project without Streamlit:
   ```
   bash run.sh
   ```

3. Access the web interface at the specified URL (typically `http://localhost:8501`) and enter the text to be classified as real or fake news.

## How to Use the Online Version

To use the software online, follow these steps:

1. Open your web browser and navigate to [Fake News Detector Web Application](https://vedoveli-fake-news.streamlit.app/).

2. Enter the text you want to check for fake news.

3. Press `Enter`.

4. The application will predict whether the provided text is real or fake news.

---

This project is for educational and informational purposes only. The accuracy of the fake news classification is not guaranteed, and users should exercise caution and critical thinking when interpreting the results.