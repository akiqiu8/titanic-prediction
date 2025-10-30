# Titanic Survival Prediction

This project builds logistic regression models in both **Python** and **R** to predict passenger survival on the Titanic using the [Kaggle dataset](https://www.kaggle.com/competitions/titanic/data).  

Both implementations are fully containerized using **Docker**, so you can reproduce results in any environment.


## Project Setup
#### 1. Clone the repository
Open your terminal and run:
```
git clone https://github.com/akiqiu8/titanic-prediction.git
```
#### 2. Download Kaggle Datasets
Go to the [Titanic Kaggle website](https://www.kaggle.com/competitions/titanic/data) and download the following files:
- ```train.csv```
- ```test.csv```
- ```gender_submission.csv```

Then create a new folder named ```data/``` under ```src/``` and place all three files inside.

Your project structure should look like this:

```
titanic-prediction/
├── src/
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   ├── python-ver/
│   │   ├── Dockerfile
│   │   └── main.py
│   │   └── requirements.txt
│   └── R-ver/
│       ├── Dockerfile
│       ├── main.R
│       └── install_packages.R
├── .gitignore
└── README.md
```

#### 3. Build the Docker Image and Run the Container
Go to the terminal and make sure you are under the ```titanic-prediction/``` directory. Then paste the following commands to build and run each version.

**Python-version:**
```
docker build -t titanic-pred-python  ./src/python-ver
docker run --rm -v "$(pwd)/src/data:/python-app/data" titanic-pred-python
```

**R-version:**
```
docker build -t titanic-pred-r  ./src/R-ver
docker run --rm -v "$(pwd)/src/data:/R-app/data" titanic-pred-r
```

This will:
- Install required packages
- Mount the ```data/``` folder into the container
- Train and evaluate the logistic regression model
- **Print training and test accuracy directly in the terminal output**
