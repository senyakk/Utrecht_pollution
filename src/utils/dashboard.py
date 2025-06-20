# RUN DASHBOARD WITH streamlit run src/utils/dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
This code provides an application for the interactive dashboard, which allows to compare
the performance of the trained models. 
"""

performance_df = pd.read_csv('results/performance.csv')

# Sidebar
st.sidebar.title("Model Monitoring Dashboard")

models = performance_df['model_name'].unique()
model_choice = st.sidebar.multiselect('Select Models for Comparison', models, default=models)

model_data = performance_df[performance_df['model_name'].isin(model_choice)]

month_names = ["April", "July", "October", "December"]
month_map = {
    "April" : 4,
    "July" : 7,
    "October" : 10,
    "December" : 12
}

month_choice = st.sidebar.selectbox("Select a month to view predictions", month_names)
predictions_df = pd.read_csv(f'results/predictions{month_map[month_choice]}.csv')

model_predictions = predictions_df[predictions_df['model_name'].isin(model_choice)]


history_dfs_O3 = {}
history_dfs_NO2 = {}

for model in model_choice:
    history_dfs_O3[model] = pd.read_csv(f'results/history_{model}_O3.csv')
    history_dfs_NO2[model] = pd.read_csv(f'results/history_{model}_NO2.csv')

# Display Model Performance
st.title("Comparison of Selected Models")

color_map = {
    "Elastic Net": "orange",
    "MLP": "navy",
    "LSTM": "firebrick"
}

col1, col2, col3 = st.columns(3)

for i, model in enumerate(model_choice):
    with [col1, col2, col3][i]:
        st.subheader(f"Performance of {model}")
        st.metric("MSE (O3)", f"{model_data.loc[model_data['model_name'] == model, 'mse_loss_O3'].values[0]:.4f}")
        st.metric("MAE (O3)", f"{model_data.loc[model_data['model_name'] == model, 'mae_loss_O3'].values[0]:.4f}")
        st.metric("MSE (NO2)", f"{model_data.loc[model_data['model_name'] == model, 'mse_loss_NO2'].values[0]:.4f}")
        st.metric("MAE (NO2)", f"{model_data.loc[model_data['model_name'] == model, 'mae_loss_NO2'].values[0]:.4f}")

# Plot for O3 predictions
st.subheader("O3 Predictions")

fig, ax = plt.subplots()

time_values = model_predictions['time'].unique()  
actual_O3 = model_predictions.groupby('time')['actual_O3'].first().reindex(time_values)

# Select number of hours
num_hours = st.sidebar.slider("Select Number of Hours to Display", min_value=1, max_value=len(time_values), value=240)
time_values = time_values[:num_hours]
actual_O3 = actual_O3.reindex(time_values)

# Plot actual O3 values
ax.plot(time_values, actual_O3, label='Actual O3', color='yellowgreen', zorder=2)

for model in model_choice:
    mdl_pred_filt = model_predictions[(model_predictions['model_name'] == model) & 
                                    (model_predictions['time'].isin(time_values))]
    ax.plot(mdl_pred_filt['time'], mdl_pred_filt['predicted_O3'],
            linestyle='--', label=f'{model}', color=color_map[model])

plt.title("Actual and Predicted Values for O3")
ax.set_xlabel('Time (hours)')
ax.set_ylabel('O3 Levels')
ax.legend()
st.pyplot(fig)
plt.clf()

# Plot for NO2 
st.subheader("NO2 Predictions")
fig, ax = plt.subplots()

time_values = model_predictions['time'].unique()  
actual_NO2 = model_predictions.groupby('time')['actual_NO2'].first().reindex(time_values)
time_values = time_values[:num_hours]
actual_NO2 = actual_NO2.reindex(time_values)

ax.plot(time_values, actual_NO2, label='Actual', color='yellowgreen')

for model in model_choice:
    mdl_pred_filt = model_predictions[(model_predictions['model_name'] == model) & 
                                    (model_predictions['time'].isin(time_values))]
    ax.plot(mdl_pred_filt['time'], mdl_pred_filt['predicted_NO2'],
            linestyle='--', label=f'{model}', color=color_map[model])

plt.title("Actual and Predicted Values for NO2")
ax.set_xlabel('Time (hours)')
ax.set_ylabel('NO2 Levels')
ax.legend()
st.pyplot(fig)
plt.clf()

# Plot: Performance Over Time for O3
st.subheader("Model Performance Over Time (O3)")
fig, ax = plt.subplots()
for model in model_choice:
    ax.plot(history_dfs_O3[model]["loss"], label=f"{model} Training Loss", color=color_map[model])
    ax.plot(history_dfs_O3[model]["val_loss"], linestyle='--', label=f"{model} Validation Loss", color=color_map[model])

ax.set_title("Training and Validation Loss for O3")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)
plt.clf()

# Performance Over Time for NO2
st.subheader("Model Performance Over Time (NO2)")
fig, ax = plt.subplots()
for model in model_choice:
    ax.plot(history_dfs_NO2[model]["loss"], label=f"{model} Training Loss", color=color_map[model])
    ax.plot(history_dfs_NO2[model]["val_loss"], linestyle='--', label=f"{model} Validation Loss", color=color_map[model])

ax.set_title("Training and Validation Loss for NO2")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)
plt.clf()
