#### Context
In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

#### Goal
To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. 

#### Data
Download the dataset in: https://www.kaggle.com/datasets/royjafari/customer-churn/data
This project utilizes a public dataset of 66,469 customers from an anonymous telecommunications company.

#### Running the project
1) Download the dataset from Kaggle link: https://www.kaggle.com/datasets/royjafari/customer-churn/data 
2) Run train.py to generate and save the model.
3) Run predict.py to initial the web service. This service continue to run while we want to predict. (see Figure_1)
4) In other Terminal, run: > python predict-test.py (in this case, the prediction to run on Windows, for this reason, it is necessary install waitress library). Two answers appear in the screen. The questions to these answers are: Does the customer churn the telecom company? What is the probabiblity of the churn? (see Figure_2)
5) Docker. To build the Docker container, use the following command: > docker build -t zoomcamp-test . . To run the container, use the following command:> docker run -it --rm -p 9696:9696 zoomcamp-test