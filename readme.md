# Motion Sensor Intrusion Detection

A detection model for detecting intrusions via motion sensor data (I count an intrusion as my brother coming into my room :p). Some of this dataset was derived via a PIR motion sensor hooked up to an ESP32 microcontroller in my room, and the rest AI generated based off that existing data. The base model chosen was an LSTM.

## Model Choice

I chose to use the LSTM because of it's great timeseries contextual understanding. Being able to take in information from previous and future cell states is exactly what I needed for this project.

## Data Pre-processing

The entire timestamp of a motion event is not required for the most part. The only case I can think of that it matters is for example whether a holiday is occurring. This model does not handle this case, however cases such as the day_of_week (0-6 encoded), hour_of_day (24 hr time), and motion_duration (seconds) are features learned by the LSTM.

Since there are a high ratio of non-intrusion to intrusion samples, I have attempted to balance the dataset by using WeightedRandomSampler which gives a weight to the two labels for all the entries, prioritizing the under-represented entries encoded with intrusion.

# Law of Large Numbers:

Dataset was of size 648 entries in the `dat.csv` file. With a batch size of 16, each epoch consists of approximately 40 batches. Over 36 epochs, the model is trained on roughly 36 * 40 = 1,440 independently sampled batches. Therefore, the `WeightedRandomSampler` which was passed to the `train_loader`, will create enough variation to even out our dataset by the end of the training. I say it needs to even out despite our `WeightedRandomSampler` because it uses a probabalistic approach to balancing the data and may not provide an exactly even split.

## Evaluation/Limitations

Early stopping was implemented to prevent the model from overfitting to the training data.

Unfortunately, the dataset I have used is still not large enough and does not have enough anomaly data to make totally acceptable inferences.



## Deployment

In progress.