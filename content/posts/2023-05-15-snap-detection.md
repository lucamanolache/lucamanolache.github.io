---
layout: post
title:  Finger Snap Detection
date:   2023-05-15 19:44:45 -0700
description: Implementation of a ConvNet model using PyTorch to detect finger snaps in audio recordings, including data collection, model architecture design, and training setup.
tags: [ml]
math: true
---

## Libraries

To detect finger snapping, I will use my favorite machine learning library `pytorch`.
One of the main reasons I am using `pytorch` is because it has `pytorch-audio`, an amazing helper librarie to work with audio.
Another reason is for the helper library, `pytorch-lightning` which makes training models easier and has automatic integration with `tensorboard`.
To install all the libraries I am using, do `pip install torch torchaudio lightning torchvision tensorboard sounddevice`.

## Dataset

When I started this project I was using [Google's audioset](https://research.google.com/audioset/).
However, this had far more data than I needed and was a pain to work with.
Additionally, I found better performance when using my own dataset which I recorded.

To make a dataset, I recommend coding a simple program to continously record 2 second intervals.
I would then snap my fingers or just do random things without snapping my fingers while the program made multiple files of recordings.
After doing this for a few minutes I would sort all my data manually into two files `snaps` and `no-snaps` to be used for training.

If you want to go by this approach, this code should be able to do this:

```python
import uuid
import sounddevice as sd

from scipy.io.wavfile import write

if __name__ == '__main__':
    fs = 44100  # Sample rate
    seconds = 2  # Duration of each recording

    while True:
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write('data/{id}.wav'.format(id=str(uuid.uuid1())), fs, recording)
```

To break down the code, `fs = 44100` sets the sample rate.
This is a represents 44.1 kHz which is a set recording rate for most streaming and consumer audio.
Different microphones might have different sample rates.
`seconds = 2` is the length of each file recorded.
When writing, we create each file name with its own uuid in order to not have duplicate names (I originally just called the files 1, 2, 3,... but when restarting the code to take a break it would overwrite old recordings).
`uuid1()` creates a uuid based on the time, other methods from uuid use different criteria to generate the uuid.

I used this to get a dataset of around 50 snaps and 50 random noises for the binary classification problem of detecting snaps or not.
One tip I have after doing this several times is to make sure you have snaps while speaking.
When I originally recorded my dataset all of my snaps had no background noise which caused major issues detecting snaps while talking.

## Model Architecture

![conv net architecture image](/images/snap-detection/model.png){: style="float: right" width="250"}

The model I am using is a simple ConvNet as shown to the right.
My original ConvNet performed far worse than the current one.
All of the changes I made were based off of the paper [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf).
In this, they suggested increasing the kernel size (for me this was from 3 -> 7).
This by itself increased my models performance by ~4%.
Additionally, they suggested using `GeLu` instead of `ReLu` as the activation function.
Similar to their discoveries, this did not increase the performance nearly as much as increasing the kernel size.
However, this paper suggested using `LinearNorm` instead of `BatchNorm`.
When doing this, my model's performance dropped significantly (around 40%).

I tried implementing the rest of the suggestions offered by the paper, including adding their modified ResNet blocks, however this lowered the performance instead (94% -> 60% with modifications).
When trying to implement their model, the performance initially dopped to ~54% which is only slightly better than randomly deciding.
Through testing, I found that by changing their `LinearNorm` to `BatchNorm` the performance could be brought up by around 10-20%.
By changing the size of each block, I managed to get to a maximam accuracy of 70%.
This, however, is far below my simple ConvNet's performance.
I believe this is because the problem of binary classification is far too simple for a large model and it most likely is overfitting.

### Implementation

To implement this model, I first define a helper module for each convolutional, norm, and activation layers.
```python
class ConvBlock(nn.Module):
    def __init__(self, inp, out, kernel=7, padding=2, stride=2):
        super().__init__()
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(inp, out, kernel_size=kernel, stride=stride, padding=padding)
        self.activation = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.bn1(x)
        return x
```

For the actual model, I am using 4 of these blocks of increasing size.

```python
class AudioClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.accuracy = torchmetrics.Accuracy(task="binary")

        # Conv layers
        conv_layers = [ConvBlock(1, 8), ConvBlock(8, 16), ConvBlock(16, 32), ConvBlock(32, 64)]
        self.conv = nn.Sequential(*conv_layers)

        # Binary Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss()

    def forward(self, x):
        x = self.conv(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)

        x = self.sigmoid(x)

        return x
```

Just as a note, I am using pytorch lightning's `pl.LightningModule` instead of the normal pytorch `nn.Module`.
If you don't want to use pytorch lightning, ignore the rest.

In order to make training this model easier, I am using pytorch lightning's built in logging to log the accuracy after every epoch.
To do this, you must overwrite the `on_validation_epoch_end` methods that get called after every validation epoch.

Additionally, for a lightning module to be trained using a pytorch lightning `Trainer`, you must overwite two more methods to dictate how every training/validation step works.
To do this, add the following functions in the `AudioClassifier` class.

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    y = y.unsqueeze(1)
    y = y.float()
    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    self.accuracy(y_hat, y)
    self.log('train_loss', loss)
    self.log('train_acc_step', self.accuracy)

    return loss

def validation_step(self, train_batch, batch_idx):
    x, y = train_batch
    y = y.unsqueeze(1)
    y = y.float()
    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    self.accuracy(y_hat, y)
    self.log('val_loss', loss)
    self.log('val_acc_step', self.accuracy)

def on_training_epoch_end(self):
    self.log('train_acc_epoch', self.accuracy)

def on_validation_epoch_end(self):
    self.log('val_acc_epoch', self.accuracy)

```

## Training

### Loading the Data

The following assumes you have your data as a `.wav` file and set up in a `data/` directory that is split further in two directories, `snap` and `no_snap`.
The approach I used to create the dataset, I created a dataframe with the location of every file along with its classification (1 for snap and 0 for no snap) which would the be used to load each audio clip on demand.
To get every file in the `snap` directory, you can use `snaps = [f for f in os.listdir("data/snap") if isfile(join("data/snap", f))]`.
Then to create the dataframe you can append all of these to a dataframe (there is probably a more efficient way to do this, but since it is only being run once, it doesn't matter).
This results in the following code:
```python
def process_file(split=0.8):
    snaps = [f for f in os.listdir("data/snap") if isfile(join("data/snap", f))]
    no_snaps = [f for f in os.listdir("data/no_snap") if isfile(join("data/no_snap", f))]

    snapping = pd.DataFrame(columns=['label', 'file'])
    for s in snaps:
        snapping.loc[len(snapping)] = [1, "data/snap/{name}".format(name=s)]
    not_snapping = pd.DataFrame(columns=['label', 'file'])
    for s in no_snaps:
        not_snapping.loc[len(not_snapping)] = [0, "data/no_snap/{name}".format(name=s)]

    snapping_train = snapping.iloc[:int(len(snapping) * split)]
    snapping_validation = snapping.iloc[int(len(snapping) * split):]
    not_snapping_train = not_snapping.iloc[:int(len(not_snapping) * split)]
    not_snapping_validation = not_snapping.iloc[int(len(not_snapping) * split):]
    df_train = pd.concat([snapping_train, not_snapping_train]).reset_index()
    df_eval = pd.concat([snapping_validation, not_snapping_validation]).reset_index()

    return df_train, df_eval

meta_data, meta_data_eval = process_file()
```

For training/validation split, I choose an 80% split, however, if you find yourself lacking data you can choose a more conservative split like 90%.

### Data Augmentation

In order to make up for the little data I had, I used the following data augementation techniques found [here]().
`time_shift` will move the audio data around so it doesn't all start/end the same.
`spectro_augment` will take a spectogram (more on that later) and put lines through it, masking parts of the data.
`spectro_gram` will generate said spectogram given raw audio.

```python
class AudioUtil:
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec
```

![augmented spectogram](/images/snap-detection/spec2.png){: style="float: right" width="250"}

Ok, spectograms.
A spectogram is a visual representation of an audio file (can be used by a ConvNet).
From what I understand, they show how much of a certain frequency appears at every point in time in an audio file.
These can end up looking quite nice in my opinion.
An example of an augmented one can be seen on the right.

### Dataloader

To load the data at a certain index, we simply look at the dataframe of files and their classifications and load it using `torchaudio`.
After loading the data, we apply the shift augmentation, turn it into a spectrogram, and add lines masking data.

```python
class SnapDataset(Dataset):
    def __init__(self, meta_data):
        self.meta_data = meta_data
        self.sr = 44100
        self.duration = 2000
        self.shift_pct = 0.2

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        path = self.meta_data['file'][idx]
        label = self.meta_data['label'][idx]

        aud = torchaudio.load(path)
        shift_aud = AudioUtil.time_shift(aud, self.shift_pct)

        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.075, n_freq_masks=3, n_time_masks=1)

        return aug_sgram, label

train_data = SnapDataset(meta_data)
test_data = SnapDataset(meta_data_eval)

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)
```

### Fitting

Now that we have our dataloaders, pytorch lightning makes training the model simple.


```python
model = AudioClassifier()

early_stop_callback = EarlyStopping(
    monitor='val_acc_epoch',
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode='max'
)

trainer = pl.Trainer(callbacks=[early_stop_callback])
trainer.fit(model, train_dataloader, test_dataloader)

model_scripted = torch.jit.script(model)
model_scripted.save('models/model.pt')
```

To prevent overfitting, we stop the model if after 5 epochs, the validation accuracy has not improved.
Finally, we save our model using torch script.

## Live Detection

This section is still a work in progress.
I have a rudementary method set up that could use plenty of improvements.

```python
while True:
    data = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, blocking=True)
    data = data[:88000]
    x = preprocess(data)
    y_hat = model(x.unsqueeze(1))
    conf = y_hat.detach().numpy()[0][0]
    print("Not a snap" if conf < 0.65 else "Snap", conf)
```

This will record an audio clip of 2 seconds and run the model we trained on it.
It will then record another, and so on.
This is not ideal as we don't get feedback as soon as we snap, we only see if a snap occured in the last 2 seconds.
Additionally, this won't tell you if you snap multiple times within 2 seconds or only once.
I plan on making improvements to this later, maybe detecting how many times a snap was heard in the past 2 seconds (prob not this though), or decreasing the recording time to the length of the average snap.
While both these have their own individual issues, it is definetly better than my current approach.
