import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
        page_title="Breast cancer detection",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

st.title("Breast cancer detection (Adaptive Boosting)")
st.write("This is a simple web app for breast cancer detection using Adaptive Boosting algorithm. The dataset used is from Kaggle Breast Cancer competition.")

@st.cache_data
def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

breast_cancer_train_link = 'https://raw.githubusercontent.com/Proggggger/Kaggle-Breast-Cancer-competition/refs/heads/main/data/train.csv'
breast_cancer_train_data = load_dataset(breast_cancer_train_link)
breast_cancer_train_data.drop('Unnamed: 32', axis=1, inplace=True)

breast_cancer_test_link = 'https://raw.githubusercontent.com/Proggggger/Kaggle-Breast-Cancer-competition/refs/heads/main/data/test.csv'
breast_cancer_test_data = load_dataset(breast_cancer_test_link)
breast_cancer_test_data.drop('Unnamed: 32', axis=1, inplace=True)

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



# y = data['concave points_mean']
# x = data['radius_mean']
# z = data['texture_worst']
# yaxis_title='Concave Points Mean',
#             xaxis_title='Radius Mean',
#             zaxis_title='Texture Worst'

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


style_string = """
<style>
.st-key-my_col_container{
padding: 15px;
border: 2px #ccc solid;
border-radius: 10px;
}
</style>
"""
st.markdown(style_string, unsafe_allow_html=True)
    





#st.dataframe(breast_cancer_data)