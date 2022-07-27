![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)

# Covid19_New_Cases_Prediction_Project :speech_balloon:
<p align="justify"> The year 2020 was a catastrophic year for humanity. Pneumonia of unknown aetiology was first reported in December 2019., since then, COVID-19 spread to the whole world and became a global pandemic. More than 200 countries were affected due to pandemic and many countries were trying to save precious lives of their people by imposing travel restrictions, quarantines, social distances, event postponements and lockdowns to prevent the spread of the virus. However, due to lackadaisical attitude, efforts attempted by the governments were jeopardised, thus, predisposing to the wide spread of virus and lost of lives.

The scientists believed that the absence of AI assisted automated tracking and predicting system is the cause of the wide spread of COVID-19 pandemic. Hence, the scientist proposed the usage of deep learning model to predict the daily COVID cases to determine if travel bans should be imposed or rescinded. </p>

## Project Descriptions :memo:
<p align="justify"> The aim of this project is <b> to create a deep learning model with only LSTM, Dense and Dropout </b> to predict new cases for Covid19. </p> 

## Project Organization :file_folder:
  ```
  ├── Datasets                                    : Contains file about the project
  ├── Statics                                     : Contains all save image (graphs/heatmap/tensorboard)
  ├── logs                                        : logs file for tensorboard
  ├── models                                      : .pkl for mms_train & test
  ├──.gitattributes                               : .gitattributes
  ├── README.md                                   : Project Descriptions
  ├── classes.py                                  : Module file in python format
  └── main_covid19.py                             : Main code file in python format
   ```

# Requirements :computer:
<p align="justify">This project is created using Spyder as the main IDE. The main frameworks used in this project are Pandas, Matplotlib, Seaborn, Scikit-learn, Tensorflow and Tensorboard.</p>

# Methodology :running:
This project contains two .py files. The training file and the module file is [main_covid19.py](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/main_covid19.py), [classes.py](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/classes.py). The flow of the projects are as follows:

  - **Step 1 - Loading the data:**
     - <p align="justify"> Data preparation is the primary step for any deep learning problem. Raw Data is retrieved from
     [dataset](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/tree/main/Datasets). This raw data consists of training and test dataset.</p>

           CSV_PATH_TRAIN = os.path.join(os.getcwd(),'Datasets','cases_malaysia_train.csv')
           CSV_PATH_TEST = os.path.join(os.getcwd(),'Datasets','cases_malaysia_test.csv')
           
           df_train = pd.read_csv(CSV_PATH_TRAIN)
           df_test = pd.read_csv(CSV_PATH_TEST)
           
           # From the dataset, the are a few rows in df[cases_new] have empty space and ?.
           # To change cases_new into numeric and change ? and empty space into NaNs
           df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')


  - **Step 2) Data Inspection:**
      - Train dataset
           
                df_train.head()
                df_train.info()
                df_train.describe().T

                # Time Series Data
                con_col_train = df_train.columns[(df_train.dtypes=='float64') | 
                                                 (df_train.dtypes=='int64')]
                print(con_col_train)

                # time Series Data Visualization (Line Plot)
                eda = EDA()
                eda.lineplot_graph(con_col_train,df_train)

                # To check NaNs value
                df_train.isna().sum()
                # From this train dataset there are 12 in cases_new and 
                # 342 NaNs in cluster_import,cluster_religious,cluster_community,
                # cluster_highRisk,cluster_education,cluster_detentionCentre, 
                # and cluster_workplace.
                # This is because there are not yet this clusters in the begining of Covid-19.
                
      - Test dataset
           
                df_test.head()
                df_test.info()
                df_test.describe().T

                 # Time Series Data
                 con_col_test = df_test.columns[(df_test.dtypes=='float64') | 
                                                (df_test.dtypes=='int64')]
                 print(con_col_test)

                 # time Series Data Visualization (Line Plot)

                 eda = EDA()
                 eda.lineplot_graph(con_col_test,df_test) 

                 # To check NaNs value
                 df_test.isna().sum()
                 # From this test dataset there are 1 NaNs in cases_new

           
  - **Step 3) Data Cleaning:**
      - Do not clean time series data unless REALLY necessary. But this data have NaNs, so we need to used intepolate. Both train and test dataset have NaNs in cases_new column.

               # Train Dataset
               df_train['cases_new'] = df_train['cases_new'].interpolate()
               # To check the NaNs
               df_train.isna().sum()
               # NaNs in cases_new in train dataset have been interpolate

               # Test Dataset
               df_test['cases_new'] = df_test['cases_new'].interpolate()
               # To check the NaNs
               df_test.isna().sum()
               # NaNs in cases_new in train dataset have been interpolate
           
   - **Step 4) Features Selection**
      - Train dataset
           
               # cases_new - Only 1 feature 
               X = df_train['cases_new']

               # Method 1: MinMaxScaler
               mms = MinMaxScaler()
               X = mms.fit_transform(np.expand_dims(X,axis=-1))

               # Save Train Min Max Scaler
               with open(MMS_SAVE_PATH, 'wb') as file:
                   pickle.dump(mms.transform,file)

               win_size = 30
               X_train = []
               y_train = []

               # produce list
               for i in range(win_size,len(X)):
                 X_train.append(X[i-win_size:i])
                 y_train.append(X[i])

               # change list to rank 3 array by using np.array
               X_train = np.array(X_train) 
               y_train = np.array(y_train)
                
      - Test dataset
        - We need 30 days of data to predict the number of covid19 cases. The test data only have 100 sample so we need to concatenate together with train data
   
              #Must to concatenate
              dataset_cat = pd.concat((df_train['cases_new'],df_test['cases_new']))

              # Method 1
              length_days = len(dataset_cat)-len(df_test)-win_size
              tot_input = dataset_cat[length_days:] 

              Xtest = mms.transform(np.expand_dims(tot_input,axis=-1))

              # Save Test Min Max Scaler
              with open(MMS_SAVE_PATH, 'wb') as file:
                  pickle.dump(mms.transform,file)

              X_test = []
              y_test = []

              # Produce list
              for i in range(win_size,len(Xtest)):
                X_test.append(Xtest[i-win_size:i])
                y_test.append(Xtest[i])

              # Change list to rank 3 array by using np.array
              X_test = np.array(X_test) 
              y_test = np.array(y_test)


    
   - **Model Development**
       - <p align="justify"> By using the model Sequential, LSTM, dropout, and Dense. </p> 
       The model can be view in [classes.py](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/classes.py) file.  
       
              md = ModelDevelopment()

              input_shape = np.shape(X_train)[1:]
              nb_class = 1
              nb_node = 32
              dropout_rate = 0.2
              activation = 'relu'

              model = md.simple_dl_model(input_shape,nb_class,nb_node,dropout_rate,activation)

              model.compile(optimizer='adam',loss='mse',metrics=['mean_absolute_percentage_error','mse'])
        
                     
      
        
   - **Model Training**
       - <p align="justify"> This model include tensorboard callbacks training the model. This model  used 500 epoch to train.</p>
       
               tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

               hist = model.fit(X_train,y_train,
                                 epochs=500,
                                 validation_data=(X_test,y_test),
                                 callbacks=(tensorboard_callback))

                                
       - <p align="justify"> The visualization of our model architecture can be presented in the figure below: </p>
       ![model](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/Statics/model-8.5_spyder.png)

   - **Model Evaluation**
      - <p align="justify"> From this sections, line graph and inverse graph were plotted and mean absolute percentage error were recorded. </p>

                me = ModelEvaluation()
                me.plot_hist_graphy(hist)

                predicted_new_cases= model.predict(X_test)

                # Plot line graph
                me.plot_line_graph(y_test, predicted_new_cases)

                # Plot inverse graph
                actual_cases = mms.inverse_transform(y_test)
                predicted_cases = mms.inverse_transform(predicted_new_cases)
                me.inverse_line_graph(actual_cases, predicted_cases)

                # Mean Absolute Percentage Error
                print(mean_absolute_percentage_error(actual_cases,predicted_cases))
                # Mean Absolute Percentage Error in percentage
                mp = MAPE()
                print("Mean_Absolute_Percentage_Error: {}".format(mp.mape(actual_cases,
                                                                       predicted_cases)))
      
  
      
# Results and Discussion :pencil:
  - **Plotting the graph**
      - <p align="justify"> Graph from tensorboard are shown below</p>
      
      ![epoch_mape](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/Statics/Web%20capture_27-7-2022_151635_colab.research.google.com.jpeg)
      ![epoch_loss](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/Statics/Web%20capture_27-7-2022_15168_colab.research.google.com.jpeg)

  - **Performance of the model and the reports as follows:**
      - <p align="justify">The mean_absolute_percentage_error is recorded at 8%. The details results is shown below. </p>
      ![MAPE](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/Statics/relu_8.5(2).JPG)
      
      - <p align="justify"> The plot line graph are shown below. </p>
      ![line](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/Statics/relu_9.7(2).png)
      
      - <p align="justify"> The plot inverse graph are shown below. </p>
      ![line](https://github.com/amaninaas/Covid19_New_Cases_Prediction_Project/blob/4c10b0d828a7fd320f01958fff6cbf57ea08263f/Statics/relu_9.7(3).png) 

  - The **MAPE recorded at 8%**, the graph shows **low loss, low mse** which indicates model is good and the graph of predicted vs Actual able to show **good-fit to the training dataset of Covid19**. **As conclusion, this model able to predict the new cases of Covid19.**


# Credits :open_file_folder:

This project is made possible by the data provided from this
[MoH-Malaysia](https://github.com/MoH-Malaysia/covid19-public)

