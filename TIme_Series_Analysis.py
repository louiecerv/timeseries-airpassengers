import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

def app():
    st.subheader('Forecasting Time Series Data with LSTM Neural Networks')
    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering)
    \nCCS 229 - Intelligent Systems
    *Department of Computer Science
    *College of Information and Communications Technology
    *##West Visayas State University##"""
    st.write(text)

    # Load the data
    df = pd.read_csv('https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv', header=0)
    with st.expander("Show Dataset"):
        st.write(df)
        st.write(df.shape)

    # Create a figure and axes using plt.subplots
    fig, ax = plt.subplots()

    # Use ax.plot to plot the data from your DataFrame
    ax.plot(df['#Passengers'])  # Assuming the data has columns for x and y values

    # (Optional) Customize your plot using ax methods
    # For example, set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Time Series Plot")

    # Display the plot
    st.pyplot(fig)    

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(df.iloc[:,1].values.reshape(-1, 1))

    # Split the data into input and output sequences
    window_size = 12
    input_data = []
    output_data = []
    for i in range(len(data_norm)-window_size):
        input_data.append(data_norm[i:i+window_size])
        output_data.append(data_norm[i+window_size])

    # Convert the data to numpy arrays
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    # Split the data into training and testing sets
    split_index = int(len(input_data) * 0.8)
    x_train = input_data[:split_index]
    y_train = output_data[:split_index]
    x_test = input_data[split_index:]

    y_test = output_data[split_index:]

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=128, input_shape=(window_size, 1)),
        tf.keras.layers.Dense(units=1)  
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    if "history" not in st.session_state:
        st.session_state.history = None

    st.sidebar.subheader("Perform LSTM Training and Forecasting")

    if st.sidebar.button("Start Training"):
        progress_bar = st.progress(0, text="Training the LSTM network, please wait...")        
        # Train the model
        history = model.fit(x_train, y_train, epochs=200, batch_size=16, validation_data=(x_test, y_test))
        st.session_state.history = history

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("LSTM Network training completed!") 

    if st.sidebar.button("Show Model Performance"):
        history = st.session_state.history

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Plot training and validation loss on the same axes
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')

        # Customize the plot using ax methods
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()  # Legend placement can be adjusted with optional arguments
        # Display the plot
        st.pyplot(fig)

        # Get the predicted values and compute the accuracy metrics
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        st.write('Train RMSE:', train_rmse)
        st.write('Test RMSE:', test_rmse)
        st.write('Train MAE:', train_mae)
        st.write('Test MAE:', test_mae)

        # Get predicted data from the model using the normalized values
        predictions = []
        for i in range(len(data_norm)-window_size):
            predbatch = model.predict(data_norm[i:i+window_size].reshape((1, window_size, 1)))
            predictions.append(predbatch)

        # Inverse transform the predictions to get the original scale
        predvalues = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))		
        predvalues = pd.DataFrame(predvalues)        
        
        # Use the model to predict the next year of data
        last_seq = data_norm[-12:] # Use the last year of training data as the starting sequence
        last_seq = np.array(last_seq)   

        preds = []
        for i in range(12):
            pred = model.predict(last_seq.reshape(1, window_size, 1))
            preds.append(pred[0])

            last_seq = np.array(last_seq)
            last_seq = np.vstack((last_seq[1:], pred[0]))       

        # Inverse transform the predictions to get the original scale
        prednext = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

        #flatten the array from 2-dim to 1-dim
        prednext = [item for sublist in prednext for item in sublist]  

        # Generate an array of datetime64 objects from January 1950 to December 1950
        months = pd.date_range(start='1950-01', end='1950-12', freq='MS')

        # Create a Pandas DataFrame with the datetime and values columns
        nextyear = pd.DataFrame({'Month': months, '#Passengers': prednext})

        time_axis = np.linspace(0, df.shape[0]-1, 12)
        time_axis = np.array([int(i) for i in time_axis])
        time_axisLabels = np.array(df.index, dtype='datetime64[D]')

        # Create a figure and axes using plt.subplots
        fig, ax = plt.subplots()
        ax = fig.add_axes([0, 0, 2.1, 2])
        ax.set_title('Comparison of Actual and Predicted Data')
        ax.plot(df.iloc[:,1].values, label='Original Dataset')
        ax.plot(list(predvalues[0]), color='red', label='Test Predictions')
        ax.set_xticks(time_axis)
        ax.set_xticklabels(time_axisLabels[time_axis], rotation=45)
        ax.set_xlabel('\nMonth', fontsize=20, fontweight='bold')
        ax.set_ylabel('No. of Passengers', fontsize=20, fontweight='bold')
        ax.legend(loc='best', prop={'size':20})
        ax.tick_params(size=10, labelsize=15)

        # Get the maximum y-value among both datasets
        max_y_value = max(df.iloc[:,1].values.max(), nextyear['#Passengers'].max())

        # Set the same y-limits for both axes
        ax.set_ylim(0, max_y_value)
        ax1 = fig.add_axes([2.3, 0, 0.4, 2])
        ax1.set_title('Projected Monthly Airline Passengers')
        ax1.plot(nextyear['#Passengers'], color='red', label='predicted')
        # Fix for xtick labels in second subplot
        ax1.set_xticklabels(nextyear['Month'].dt.strftime('%Y-%m'), rotation=45)  # Use strftime for formatting                
        ax1.set_xlabel('Month', fontsize=20, fontweight='bold')
        ax1.set_ylabel('No. of Passengers', fontsize=20, fontweight='bold')
        ax1.tick_params(size=10, labelsize=15) 

        # Set the same y-limits for both axes
        ax1.set_ylim(0, max_y_value)
        st.pyplot(fig)

    st.write("© 2024 West Visayas State university. All rights reserved.")
if __name__ == '__main__':
    app()   


