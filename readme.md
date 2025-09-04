# Motion Sensor Intrusion Detection

A detection model for detecting intrusions via motion sensor data (I count an intrusion as my brother coming into my room :p). Some of this dataset was derived via a PIR motion sensor hooked up to an ESP32 microcontroller in my room, and the rest AI generated based off that existing data. The base model chosen was an LSTM.

## Model Choice

I chose to use the LSTM because of it's great timeseries contextual understanding. Being able to take in information from previous and future cell states is exactly what I needed for this project.

## Data Pre-processing

The entire timestamp of a motion event is not required for the most part. The only case I can think of that it matters is for example whether a holiday is occurring. This model does not handle this case, however cases such as the day_of_week (0-6 encoded), hour_of_day (24 hr time), and motion_duration (seconds) are features learned by the LSTM.

Since there are a high ratio of non-intrusion to intrusion samples, I have attempted to balance the dataset by using WeightedRandomSampler which gives a weight to the two labels for all the entries, prioritizing the under-represented entries encoded with intrusion.

# Law of Large Numbers / Central Limit Theorem:

Epochs ran on the last trial was 36, which is greater than 30, which means we can assume the WeightedRandomSampler which was passed to the train_loader, will create enough variation over the 36 epochs to even out our dataset by the end of the training.

## Evaluation


Epoch 36, Val Loss: 0.0072, Val Acc: 100.00%, Val F1: 1.0000
(Accuracy calculation used rounded values output from the model, these were float32s so they were in the continuous range from 0 - 1)

Early stopping was implemented to prevent the model from overfitting to the training data.

## Deployment

In progress.