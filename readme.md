# Motion Sensor Intrusion Detection

A detection model for detecting intrusions via motion sensor data (I count an intrusion as my brother coming into my room :p). Some of this dataset was derived via a PIR motion sensor hooked up to an ESP32 microcontroller in my room, and the rest AI generated based off that existing data. The base model chosen was an LSTM.

## Model Choice

I chose to use the LSTM because of it's great timeseries contextual understanding. Being able to take in information from previous and future cell states is exactly what I needed for this project.

## Data Pre-processing

The entire timestamp of a motion event is not required for the most part. The only case I can think of that it matters is for example whether a holiday is occurring. This model does not handle this case, however cases such as the day_of_week (0-6 encoded), hour_of_day (24 hr time), and motion_duration (seconds) are features learned by the LSTM.

Once the dataset has been split (using slicing not sklearn `train_test_split` since that internally shuffles data to maintain temporal data), I sequenced the timeseries data using a sliding window for sampling by the future dataloader.

### BUT

Since there is a high ratio of non-intrusion to intrusion samples, I have attempted to balance the dataset by using WeightedRandomSampler which gives a weight to the two labels for all the sequences, prioritizing the under-represented sequences with minority labels.

## Training

Just your basic train test loop that prints some evaluation stats.

Early stopping was implemented to prevent the model from overfitting to the training data.


## Evaluation/Limitations

Unfortunately, from the f1 score and just obvious logic, there is an insufficient amount of balanced data to produce >90% accuracy in detections. This may be due to so much repeated sampling of the same data since my data was very much imbalanced with 169 entries with the encoded label for `intrusion` and 479 with the encoded label for `non-intrusion`.

### The Law of Large Numbers:

Dataset was of size 648 entries in the `dat.csv` file. Of this, 508 were used to train this model. With a batch size of 16, each epoch consisting of 31.75 batches. The model reached 18 epochs, the before early_stopping due in fear of overfitting.

As a result, there are approximately 18 * 31.75 = 571.5 total independent samplings.

This is relatively small N, however, and the `WeightedRandomSampler` which was passed to the `train_loader`, will likely not be able to create enough variation to even out our dataset by the end of the training. I say it needs to even out despite our `WeightedRandomSampler` because it uses a probabalistic approach to balancing the data and may not provide an exactly even split.

## Deployment

In progress.