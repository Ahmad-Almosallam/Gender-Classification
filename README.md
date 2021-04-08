# Gender-Classification

Gender Classification through voice, after searching I found amazing database for audio clips from audiobooks recordings at: http://www.openslr.org/12/.
This dataset contains many gigabytes of clean data in ".flac" files. I used the train-clean-100.tar.gz [6.3G]. 

## Project
1- Building a gender classifier model

2- Developing a GUI to use the gender classifier


### Gender classifier

As we know audio at the end are numbers, so I extracted some features from the audio and stored it in a numpy array and then saved it as a file. 
After that, I have done preprocessing steps for the data through StanderScaler and LabelEncoder. 

Then, building models to test out the data. I have used a 6 algorithms such as SVC, RandomForestClassifier, GradientBoostingClassifier, and so on. However, the results were close 
to each other. But here are the best three models.

1- RandomForestClassifier

2- GradientBoostingClassifier

3- SVC


At the end, I used Pickle to use the model in the Web demo.

### Web Demo

I used Flask to do a web demo.
