# Deep-Utils
 A collection of .bat files designed to make the process of creating machine learning models **extremly easy** using AutoKeras

 ## Usage
 The process of model creation with Deep-Utils is divided into 3 parts:
 ### **Data Preperation**
 Every model starts with data preperations.  
 Deep-Utils requires a `data.csv` file, which is organized to suit the type of model you are creating.  
 1. for Image Classification it will look like this:  
 `Path to image (String), image label (Int)` 
 2. for Image Regression it will look like this:  
   `Path to Image (String), Value 1 (Int/Float), Value 2 (Int/Float), etc...`

Before you begin the next step, make sure to run `operations/0. delete_previous_progress.bat` to make sure you are ready for new model creation.  
  
*Note: the path to the image is from the 'data' folder. e.g. IMGS/img0.png*  
*Note: support for more data types is coming in the future*  

 ### **Model Training**
 Model Training is the main step of the creation process.  
 It starts by setting the parameters in `operations/parameters.json`.  
 The parameters are:  
    1.`"maxDataSize":3000,` *This is the amount of samples to load from `data.csv`.*  
    2.`"trainTestRatio":0.7,` *This defines how the dataset is split, in this case 70% goes to training.*  
    3.`"imgSizeRatio":0.5,` *This is the scale multiplier of loaded images, if you do not care about the scale, or not using images, you can leave this at 1.0 .*  
    4.`"trainEpochs":50,` *This is the maximum amount of epochs the model will train.*  
    5.`"maxTrials":2` *This is the amount of models to look through, in this case the program will try 2 different models with different hyper-parameters. The higher the value the slower the training, but the model will be suited better for the data.*  
After you've set your parameters, start `operations/1. train_modelType.bat` with `modelType` replaced by the type you want to create. The model will then start training and will notify you when it is done.
 ### **Model Evaluation**
Evaluation is the shortest and simplest of steps. Start `operations/2. evaluate_model.bat`, the program will display the Loss and Accuracy on unseen data. If you are satisfied with the results you can finish the process here, if not you should go back to **Model Training**.  
You can also run the model on a single unseen sample and display the results using `operations/3. preview_model.bat`.  
If you want to get a summary of the built model, run `operations/4. summarize_model.bat`.  

**Your Final Model should be waiting for you in the main folder!**  
  
  
  *Note: support for more model types will be added soon*