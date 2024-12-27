# Hackaton models

## Random Forest Classifier

The approach was test different models and hyperparameters to find the best model for the dataset. The Random Forest Classifier was the best model for the dataset.

Kaggle Dataset: https://www.kaggle.com/datasets/gorororororo23/plant-growth-data-classification

Data length: 192 data points

Data features:
- Soil_Type: Type of soil where the plant is growing. 'loam' | 'sandy' | 'clay'
- Sunlight_Hours: Number of hours the plant is exposed to sunlight daily. Float
- Water_Frequency: Frequency of watering the plant. 'bi-weekly' | 'weekly' | 'daily'
- Fertilizer_Type: Type of fertilizer used for the plant. 'chemical' | 'organic' | 'none'
- Temperature: Average temperature in the plant's environment. Float
- Humidity: Average humidity in the plant's environment. Float

Data labels:
- Growth_Milestone: Indicates whether the plant has reached a specific growth milestone. True | False

Final accuracy: 0.75

## CNN Classifier

The approach was to use a Convolutional Neural Network to classify the images. The model was trained using the training set and tested using the test set. The model was able to classify the images with an accuracy of 0.45.

Kaggle Dataset: https://www.kaggle.com/datasets/marquis03/plants-classification

Data length: 700 images * 30 classes at training set. 200 images * 30 classes at test set.

Data classes:
- aloevera
- banana
- bilimbi
- cantaloupe
- cassava
- coconut
- corn
- cucumber
- curcuma
- eggplant
- galangal
- ginger
- guava
- kale
- longbeans
- mango
- melon
- orange
- paddy
- papaya
- peperchili
- pineapple
- pomelo
- shallot
- soybeans
- spinach
- sweetpotatoes
- tobacco
- waterapple
- watermelon

Final accuracy: 0.45
