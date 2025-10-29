import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OrdinalEncoder
from sklearn.calibration import calibration_curve
from sklearn.ensemble import AdaBoostClassifier
import pickle
from urllib.request import urlopen

st.set_page_config(
        page_title="Breast cancer detection",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

st.title("Breast cancer detection (Adaptive Boosting)")
st.write("This is a simple web app for breast cancer detection using Adaptive Boosting algorithm. The dataset used is from Kaggle Breast Cancer competition.")
st.write("You can explore the dataset, view model training results, and make manual predictions by entering feature values.")
st.write("Developed by [Ivan Burmaka](https://github.com/Proggggger)")
st.write("Explore the complete code and resources in the [GitHub repository](https://github.com/Proggggger/Kaggle-Breast-Cancer-competition), including training algorithms and data preprocessing steps.")
st.write("Also check out the [Kaggle competition page](https://www.kaggle.com/competitions/breast-cancer-detection/code) for more details.")
@st.cache_data
def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

@st.cache_resource
def load_model():
    url = "https://github.com/Proggggger/Kaggle-Breast-Cancer-competition/raw/refs/heads/main/models/ada_model_.pkl"
    with urlopen(url) as f:
        return pickle.load(f)

def plot_confusion_matrix(y_true, y_pred):
    """
    Creates an interactive confusion matrix visualization
    """
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create text annotations
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=str(cm[i, j]),
                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black"),
                    showarrow=False
                )
            )
    
    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        annotation_text=cm.astype(str),
        colorscale='Blues',
        showscale=True
    )
    
    # Update layout
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        xaxis=dict(tickmode='array', tickvals=labels),
        yaxis=dict(tickmode='array', tickvals=labels),
        margin=dict(l=100, r=100, b=100, t=100),
        annotations=annotations
    )
    
    return fig


def plot_calibration_curve(model, X, y, n_bins=10):
    """
    Creates an interactive calibration curve visualization
    """
    enc = OrdinalEncoder()
    y = enc.fit_transform(y.values.reshape(-1, 1)).astype(int)
    y = np.where(y == enc.categories_[0].tolist().index('M'), 1, 0)
    # Get predicted probabilities for positive class
    prob_pos = model.predict_proba(X)[:, 1]
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y, prob_pos, n_bins=n_bins, strategy='quantile'
    )
    
    # Create perfect calibration line
    perfect_line = np.linspace(0, 1, 100)
    
    # Create figure
    fig = go.Figure()
    
    # Add calibration curve
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name='Model Calibration',
        marker=dict(size=10, color='blue'),
        line=dict(width=3)
    ))
    
    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=perfect_line,
        y=perfect_line,
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title='Calibration Curve',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


breast_cancer_train_link = 'https://raw.githubusercontent.com/Proggggger/Kaggle-Breast-Cancer-competition/refs/heads/main/data/train.csv'
breast_cancer_train_data = load_dataset(breast_cancer_train_link)
breast_cancer_train_data.drop('Unnamed: 32', axis=1, inplace=True)

breast_cancer_test_link = 'https://raw.githubusercontent.com/Proggggger/Kaggle-Breast-Cancer-competition/refs/heads/main/data/test.csv'
breast_cancer_test_data = load_dataset(breast_cancer_test_link)
breast_cancer_test_data.drop('Unnamed: 32', axis=1, inplace=True)

pretrained_model = load_model()

def create_3d_plot(data):
    fig = px.scatter_3d(
        data,
        x='radius_mean',
        y='concave points_mean',
        z='texture_worst',
        color='diagnosis',
        color_discrete_map={'M': 'orange', 'B': 'blue'},
        title='3D Scatter Plot of Breast Cancer Features',
        labels={'diagnosis': 'Diagnosis'}  # Optional: customize legend title
    )
    # Customize axis titles if needed
    fig.update_layout(
        scene=dict(
            xaxis_title='Radius Mean',
            yaxis_title='Concave Points Mean',
            zaxis_title='Texture Worst'
        )
    )
    return fig
# Initial 3D plot

fig = create_3d_plot(breast_cancer_train_data)


with st.container(key="my_col_container"):
    st.header('Data review')
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(key="my_col1_container"):
            dataset_choice = st.radio('Select Dataset', ['Train', 'Test'], index=0, horizontal=True)

    with col2:
        show_hide_clicked = st.button('Show/Hide DataFrame', key="my_button_key_df")

    with col3:
        show_hidep_clicked = st.button('Show/Hide Plot', key="my_button_key_plt")
        st.write('(train only)')

    if show_hide_clicked:
        st.session_state.show_df = not st.session_state.get('show_df', False)

    if show_hidep_clicked:
        st.session_state.show_plt = not st.session_state.get('show_plt', False)    

    if st.session_state.get('show_df', False):
        st.markdown('<style> .st-key-my_button_key_df button { background-color: #aaa !important; color: white; } </style>', unsafe_allow_html=True)
    else:
        st.markdown('<style> .st-key-my_button_key_df button { background-color: green !important; color: white; } </style>', unsafe_allow_html=True)

    if st.session_state.get('show_plt', False):
        st.markdown('<style> .st-key-my_button_key_plt button { background-color: #aaa !important; color: white; } </style>', unsafe_allow_html=True)
    else:
        st.markdown('<style> .st-key-my_button_key_plt button { background-color: green !important; color: white; } </style>', unsafe_allow_html=True)

    if dataset_choice == 'Train':
        selected_data = breast_cancer_train_data
    else:
        selected_data = breast_cancer_test_data
    fig = create_3d_plot(breast_cancer_train_data)    

    if st.session_state.get('show_df', False):
        st.dataframe(selected_data)

    if st.session_state.get('show_plt', False):
        st.plotly_chart(fig, use_container_width=True)


X = breast_cancer_train_data.drop(['id', 'diagnosis'], axis=1)
y = breast_cancer_train_data['diagnosis']
pred = pretrained_model.predict(X)

with st.container(key="my_col_container2"):
    st.header("Model training results section")
    st.write(f"Model accuracy on training data: {np.mean(pred == y):.4f}")
    st.write("(on training data only due to lack of test labels)")
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        show_hide_confusion = st.button('Show/Hide Confusion Matrix', key="my_button_key_conf")
    with col2_2:    
        show_hide_calibration = st.button('Show/Hide Calibration curve', key="my_button_key_cal")

        
    if show_hide_calibration:
        st.session_state.show_cal = not st.session_state.get('show_cal', False)
    if show_hide_confusion:
        st.session_state.show_conf = not st.session_state.get('show_conf', False)



    plot_cm = plot_confusion_matrix(y, pred)
    if st.session_state.get('show_conf', False):
        st.subheader('Confusion Matrix')
        st.plotly_chart(plot_cm, use_container_width=True)


    cal_fig = plot_calibration_curve(pretrained_model, X, y)
    if st.session_state.get('show_cal', False):
        st.subheader('Model Calibration')
        st.plotly_chart(cal_fig, use_container_width=True)

    if st.session_state.get('show_conf', False):
        st.markdown('<style> .st-key-my_button_key_conf button { background-color: #aaa !important; color: white; } </style>', unsafe_allow_html=True)
    else:
        st.markdown('<style> .st-key-my_button_key_conf button { background-color: green !important; color: white; } </style>', unsafe_allow_html=True)

    if st.session_state.get('show_cal', False):
        st.markdown('<style> .st-key-my_button_key_cal button { background-color: #aaa !important; color: white; } </style>', unsafe_allow_html=True)
    else:
        st.markdown('<style> .st-key-my_button_key_cal button { background-color: green !important; color: white; } </style>', unsafe_allow_html=True)


style_string = """
<style>
.st-key-my_col_container, .st-key-my_col_container2 {
padding: 15px;
border: 2px #ccc solid;
border-radius: 10px;
}
</style>
"""
st.markdown(style_string, unsafe_allow_html=True)
    





# Calculate feature boundaries for input validation
# Extract feature columns (excluding id and diagnosis)
feature_columns = [col for col in breast_cancer_train_data.columns if col not in ['id', 'diagnosis']]
# Calculate min, max, and mean values for each feature to set input boundaries and default values
feature_min_values = breast_cancer_train_data[feature_columns].min()
feature_max_values = breast_cancer_train_data[feature_columns].max()
feature_mean_values = breast_cancer_train_data[feature_columns].mean()

# New section for manual prediction input
with st.container(key="prediction_container"):
    st.header('Manual Prediction')
    st.write("Enter the values for each feature to get a prediction from the model. (Values ranges are limited to training data ranges)")
    
    # Create three columns for better organization of input fields
    col1, col2, col3 = st.columns(3)
    
    # Dictionary to store user input values
    input_data = {}
    
    # First 10 features (mean values) in first column
    with col1:
        st.subheader("Mean Features")
        for i, feature in enumerate(feature_columns[:10]):
            # Create number input with min/max boundaries and mean as default value
            input_data[feature] = st.number_input(
                label=f"{feature.replace('_', ' ').title()} (Min: {feature_min_values[feature]:.4f}, Max: {feature_max_values[feature]:.4f})",
                min_value=float(feature_min_values[feature]),
                max_value=float(feature_max_values[feature]),
                value=float(feature_mean_values[feature]),
                step=0.01,
                format="%.4f",
                key=f"input_{feature}"
            )
    
    # Next 10 features (standard error values) in second column
    with col2:
        st.subheader("Std Error Features")
        for i, feature in enumerate(feature_columns[10:20]):
            # Create number input with min/max boundaries and mean as default value
            input_data[feature] = st.number_input(
                label=f"{feature.replace('_', ' ').title()} (Min: {feature_min_values[feature]:.4f}, Max: {feature_max_values[feature]:.4f})",
                min_value=float(feature_min_values[feature]),
                max_value=float(feature_max_values[feature]),
                value=float(feature_mean_values[feature]),
                step=0.01,
                format="%.4f",
                key=f"input_{feature}"
            )
    
    # Last 10 features (worst values) in third column
    with col3:
        st.subheader("Worst Features")
        for i, feature in enumerate(feature_columns[20:]):
            # Create number input with min/max boundaries and mean as default value
            input_data[feature] = st.number_input(
                label=f"{feature.replace('_', ' ').title()} (Min: {feature_min_values[feature]:.4f}, Max: {feature_max_values[feature]:.4f})",
                min_value=float(feature_min_values[feature]),
                max_value=float(feature_max_values[feature]),
                value=float(feature_mean_values[feature]),
                step=0.01,
                format="%.4f",
                key=f"input_{feature}"
            )
    
    # Prediction button - triggers prediction when clicked
    if st.button('Make Prediction', key="predict_button"):
        # Convert input data to DataFrame for model prediction
        input_df = pd.DataFrame([input_data])
        
        # Make prediction using the pretrained model
        prediction = pretrained_model.predict(input_df)[0]
        prediction_proba = pretrained_model.predict_proba(input_df)[0]
        
        # Display prediction result
        st.subheader('Prediction Result')
        
        # Create two columns for result display
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            # Display prediction with appropriate styling
            if prediction == 'M':
                st.error(f"Prediction: **Malignant**")
            else:
                st.success(f"Prediction: **Benign**")
        
        with result_col2:
            # Display prediction probabilities
            # Handle class order to correctly assign probabilities
            benign_prob = prediction_proba[0] if pretrained_model.classes_[0] == 'B' else prediction_proba[1]
            malignant_prob = prediction_proba[1] if pretrained_model.classes_[1] == 'M' else prediction_proba[0]
            
            st.write("Prediction Probabilities:")
            st.write(f"Benign: {benign_prob:.4f} ({benign_prob*100:.2f}%)")
            st.write(f"Malignant: {malignant_prob:.4f} ({malignant_prob*100:.2f}%)")
        
        # Create a visual gauge for probability visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = malignant_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Malignant Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Create 3D scatter plot with manual point visualization
        st.subheader('3D Visualization')
        st.write("Your input point (green) compared to training data points")
        
        # Create a copy of training data for visualization
        viz_data = breast_cancer_train_data.copy()
        
        # Add the manual input point to the data with a unique identifier
        manual_point = input_data.copy()
        manual_point['diagnosis'] = 'Your Input'  # Use a unique identifier for coloring
        viz_data = pd.concat([viz_data, pd.DataFrame([manual_point])], ignore_index=True)
        
        # Create the 3D scatter plot
        fig_3d = px.scatter_3d(
            viz_data,
            x='radius_mean',
            y='concave points_mean',
            z='texture_worst',
            color='diagnosis',
            color_discrete_map={'M': 'orange', 'B': 'blue', 'Your Input': '#b6e680'},
            title='3D Scatter Plot: Training Data + Your Input',
            labels={'diagnosis': 'Diagnosis'},
            hover_data=['diagnosis']
        )
        
        # Set default marker size for all points
        fig_3d.update_traces(
            selector=dict(mode='markers'),
            marker=dict(size=5)
        )
        
        # Make the manual point larger and more visible with black border
        fig_3d.update_traces(
            selector=dict(marker_color='#b6e680'),
            marker=dict(size=12, line=dict(width=2, color='black'))
        )
        
        # Customize axis titles and legend
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Radius Mean',
                yaxis_title='Concave Points Mean',
                zaxis_title='Texture Worst'
            ),
            legend=dict(
                title="Diagnosis",
                itemsizing='constant'
            )
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)

# Add styling for the prediction container
prediction_style = """
<style>
.st-key-prediction_container {
    padding: 15px;
    border: 2px #4CAF50 solid;
    border-radius: 10px;
    margin-top: 20px;
}
</style>
"""
st.markdown(prediction_style, unsafe_allow_html=True)
#st.dataframe(breast_cancer_data)