# Arabic_sign_language_improved


This is an improved version of an old arabic sign language project I had made for a semester project, and it uses a dataset i found on kaggle

We use a CNN to make a classification model for 31 possible classes in arabic sign language
The classes are: Ain, Al, Alef, Beh, Dad, Dal, Feh, Ghain, Hah, Heh, Jeem, Kaf, Khah, Laa, Lam, Meem, Noon, Qaf, Reh, Sad, Seen, Sheen, Tah, Teh, Teh_Marbuta, Thal, Theh, Waw, Yeh, Zah, Zain

We preprocess the images, use tensorflow to make a CNN model, train it on processed dataset, and use a small interface to test it

Libraries used: Tensorflow, CouchDB, tkinter, PIL, rembg

Essential scripts:
- preprocessing.py: goes through every image in the original dataset folder, removes their background, resizes them, makes them monochrome and keeps only the edges (which would end up being just the edges of a hand)
- model.py: uses tensorflow to build a CNN made out of multiple layers
- train.py: loads the dataset, normalizes and augments some of it, compiles and generates a model using the parameters from model.py and config.json, trains the model with features such as model checkpoints and autostop if the model consistently stops learning
- interface.py: contains a basic interface that allows you to select images and have the model make a prediction on, it applies the preprocessing from preprocessing.py to the images before attempting to predict what they are

The best model we've managed to make reaches about a 70% accuracy

Possible ideas to expand the project: add a webcam and/or video functionality, play around with the preprocessing more as to get more accurate results.
