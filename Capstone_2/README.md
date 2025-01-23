### Context
Heart disease is broad term used for diseases and conditions affecting the heart and circulatory system. They are also referred as cardiovascular diseases. It is a major cause of disability all around the world. 
There are several different types and forms of heart diseases. In this dataset is listed various heart diseases associated to different features. 

### Goal
Detecting the probability of person that will be affected by a heart problem or not. From this probablity, we will predict if a person can or not suffer some cardiovascular disease from his/her values in each feature studied.

### Data
Download the dataset in: https://www.kaggle.com/datasets/belsonraja/heart-disease-prediction/data

Dictionary of the terms used in the dataset:

- age: age in years
- sex: sex [1 = male, 0 = female]
- cp: chest pain type [Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic]
- trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholestoral in mg/dl
- fbs: (fasting blood sugar > 120 mg/dl) [1 = true; 0 = false]
- restecg: resting electrocardiographic results [Value 0: normal, Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- thalach: maximum heart rate achieved
- exang: exercise induced angina [1 = yes, 0 = no]
- oldpeak = ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment [Value 0: upsloping, Value 1: flat, Value 2: downsloping]
- ca: number of major vessels (0-3) colored by flourosopy
- thal: [0 = error (in the original dataset 0 maps to NaN's), 1 = fixed defect, 2 = normal, 3 = reversable defect]
- target (the lable): [0 = no disease, 1 = disease]

### Running the project

#### Preparing the Environment
1) Clone the repository: 

git clone https://github.com/sanchis135/MLZoomcamp2024.git

cd MLZoomcamp2024/Capstone_1

2) Create a virtual environment and activate it:

python -m venv .venv
& .\.venv\Scripts\activate  

3) Install the required dependencies:

pip install -r requirements.txt

#### Training Model
4) Train the model. Run train.py to generate and save the model:

python train.py

#### Running the Prediction Service
5) Run predict.py to initial the web service. This service continue to run while we want to predict. 

python predict.py

6) In other Terminal, run the test: 

python predict-test.py 

(In this case, the prediction was running on Windows, for this reason, it was necessary install waitress library).

#### Docker Deployment

7) Install Pipenv if you haven't already:

pip install pipenv

8) Activate the environment:

pipenv shell

9) Build the Docker image:

docker build -t patient-test .

10) Run the Docker container:

docker run -it --rm -p 9696:9696 patient-test
