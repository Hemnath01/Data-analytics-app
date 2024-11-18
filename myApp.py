import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Page configuration
st.set_page_config(
    page_title="Daily Use Analytics Portal",
    page_icon="🔹"
)

# Initialize session states
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False

# Title
st.title("My :red[Data] :blue[Analytics] Portal")
st.subheader(":grey[Explore data with ease]", divider="orange")

# File uploader
file = st.file_uploader("Drop CSV", type="csv", key="file_uploader_1")

# Check if file is uploaded
if file is not None and not st.session_state.file_uploaded:
    if file.name.endswith("csv"):
        st.session_state.data = pd.read_csv(file)
    else:
        st.session_state.data = pd.read_excel(file)
    st.session_state.file_uploaded = True
    st.info("File is successfully uploaded", icon="🚨")

if st.session_state.data is not None:
    # Display the current state of the data
    st.dataframe(st.session_state.data)
    
    st.subheader("Basic information of the dataset", divider="green")
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Top and Bottom Rows", "Data Types", "Columns"])

    with tab1:
        st.subheader(":grey[Statistical summary of the dataset]")
        st.write(f'There are {st.session_state.data.shape[0]} rows and {st.session_state.data.shape[1]} columns in the dataset')
        st.dataframe(st.session_state.data.describe())
        
    with tab2:
        st.subheader(":grey[Top rows]")
        toprows_input = st.text_input("Enter the number of top rows you want:", value="5", key="topinput")
        if st.button("Show Top Rows"):
            if toprows_input.isdigit():
                toprows = int(toprows_input)
                toprows = min(toprows, st.session_state.data.shape[0])
                st.dataframe(st.session_state.data.head(toprows))
            else:
                st.error("Please enter a valid number.")

        st.subheader(":grey[Bottom rows]")
        bottomrows_input = st.text_input("Enter the number of bottom rows you want:", value="5", key="bottominput")
        if st.button("Show Bottom Rows"):
            if bottomrows_input.isdigit():
                bottomrows = int(bottomrows_input)
                bottomrows = min(bottomrows, st.session_state.data.shape[0])
                st.dataframe(st.session_state.data.tail(bottomrows))
            else:
                st.error("Please enter a valid number.")
                
    with tab3:
        st.subheader(":grey[Data types of column]")
        st.dataframe(st.session_state.data.dtypes)
        
    with tab4:
        st.subheader(":grey[Column names in dataset]")
        st.write(list(st.session_state.data.columns))

    st.subheader("Column values to count", divider="green")
    
    with st.expander("Value Count"):
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox("Choose column name", options=list(st.session_state.data.columns))
        with col2:
            toprows = st.number_input("Top rows", min_value=1, step=1)
        count = st.button("Count")
        if count:
            result = st.session_state.data[column].value_counts().reset_index().head(toprows)
            st.dataframe(result)

            st.subheader("Visualization", divider="grey")
            fig = px.bar(data_frame=result, x=column, y="count", text="count", template="plotly_dark")
            st.plotly_chart(fig)
            fig = px.line(data_frame=result, x=column, y="count", text="count", template="plotly_dark")
            st.plotly_chart(fig)
            fig = px.pie(data_frame=result, names=column, values="count", template="plotly_dark")
            st.plotly_chart(fig)

    st.subheader('Groupby : Simplify your data analysis', divider='green')
    st.write('The groupby lets you summarize data by specific categories and groups')
    
    with st.expander("Groupby"):
        col1, col2, col3 = st.columns(3)
        with col1:
            groupby_cols = st.multiselect("Choose column to groupby", options=list(st.session_state.data.columns))
        with col2:
            operation_col = st.selectbox("Choose column for operation", options=list(st.session_state.data.columns))
        with col3:
            operation = st.selectbox("Choose operation", options=["mean", "median", "max", "min", "count", "sum"])
        
        if groupby_cols:
            result = st.session_state.data.groupby(groupby_cols).agg(
                newcol=(operation_col, operation)
            ).reset_index()
            st.dataframe(result)
            
            st.subheader(":grey[Data Visualization]", divider="grey")
            graph = st.selectbox("Choose your graphs", options=['line', 'bar', 'scatter', 'pie', 'sunburst'])
            
            if graph == 'line':
                x_axis = st.selectbox('Choose X axis', options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis', options=list(result.columns))
                color = st.selectbox('Color Information', options=[None] + list(result.columns))
                fig = px.line(data_frame=result, x=x_axis, y=y_axis, color=color, markers="o")
                st.plotly_chart(fig)
            
            elif graph == 'bar':
                x_axis = st.selectbox('Choose X axis', options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis', options=list(result.columns))
                color = st.selectbox('Color Information', options=[None] + list(result.columns))
                facet_col = st.selectbox("Column Information", options=[None] + list(result.columns))
                fig = px.bar(data_frame=result, x=x_axis, y=y_axis, color=color, facet_col=facet_col, barmode="group")
                st.plotly_chart(fig)

            elif graph == 'scatter':
                x_axis = st.selectbox('Choose X axis', options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis', options=list(result.columns))
                color = st.selectbox('Color Information', options=[None] + list(result.columns))
                size = st.selectbox("Size Column", options=[None] + list(result.columns))
                fig = px.scatter(data_frame=result, x=x_axis, y=y_axis, color=color, size=size)
                st.plotly_chart(fig)

            elif graph == 'pie':
                names = st.selectbox('Choose Labels', options=list(result.columns))
                values = st.selectbox('Choose Numerical Values', options=list(result.columns))
                fig = px.pie(data_frame=result, values=values, names=names)
                st.plotly_chart(fig)

            elif graph == 'sunburst':
                path = st.multiselect('Choose Your Path (Hierarchy)', options=list(result.columns))
                values = st.selectbox('Choose Numerical Values for Sunburst', options=list(result.columns))
                fig = px.sunburst(data_frame=result, path=path, values=values)
                st.plotly_chart(fig)

    st.header(":blue[Apply Machine Learning]", divider="green")
    st.subheader(":grey[Choose Problem Type]")

    problem_type = st.radio("Select One Machine Learning Model", ["Regression", "Classification"])

    st.subheader(":grey[Preprocessing]")

    with st.expander("Handle Missing Values"):
        col_missing = st.selectbox("Select column with missing values", options=list(st.session_state.data.columns))
        missing_method = st.radio(
            "Choose Method",
            ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows", "Drop Column"],
            key="missing_method"
        )
        
        if st.button("Apply Missing Value Handling", key="apply_missing"):
            temp_data = st.session_state.data.copy()
            
            if missing_method == "Fill with Mean":
                temp_data[col_missing] = temp_data[col_missing].fillna(temp_data[col_missing].mean())
            elif missing_method == "Fill with Median":
                temp_data[col_missing] = temp_data[col_missing].fillna(temp_data[col_missing].median())
            elif missing_method == "Fill with Mode":
                temp_data[col_missing] = temp_data[col_missing].fillna(temp_data[col_missing].mode()[0])
            elif missing_method == "Drop Rows":
                temp_data = temp_data.dropna(subset=[col_missing])
            elif missing_method == "Drop Column":
                temp_data = temp_data.drop(col_missing, axis=1)
            
            st.session_state.data = temp_data
            st.success("Missing values handled successfully.")
            st.write("Updated Data Head:")
            st.write(st.session_state.data.head())

    with st.expander("Normalize or Standardize Data"):
        cols_to_scale = st.multiselect(
            "Select Columns to Scale",
            st.session_state.data.select_dtypes(include=np.number).columns,
            key="scale_cols"
        )
        scale_type = st.radio(
            "Choose Scaling Method",
            ["Min-Max Scaling", "Standard Scaling"],
            key="scaling_method"
        )
        
        if st.button("Apply Scaling", key="apply_scaling"):
            if cols_to_scale:
                temp_data = st.session_state.data.copy()
                if scale_type == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                temp_data[cols_to_scale] = scaler.fit_transform(temp_data[cols_to_scale])
                st.session_state.data = temp_data
                st.success("Scaling applied successfully.")
                st.write("Updated Data Head:")
                st.write(st.session_state.data.head())
            else:
                st.warning("No columns were selected for scaling")

    with st.expander("Encode Categorical Data"):
        cat_cols = st.multiselect(
            "Select Categorical Columns to Encode",
            st.session_state.data.select_dtypes(include="object").columns,
            key="encode_cols"
        )
        encoding_method = st.radio(
            "Choose Encoding Method",
            ["One-Hot-Encoding", "Label Encoding"],
            key="encoding_method"
        )
        
        if st.button("Apply Encoding", key="apply_encoding"):
            temp_data = st.session_state.data.copy()
            try:
                if encoding_method == "One-Hot-Encoding":
                    temp_data = pd.get_dummies(temp_data, columns=cat_cols)
                else:  # Label Encoding
                    le = LabelEncoder()
                    for col in cat_cols:
                        temp_data[col] = le.fit_transform(temp_data[col])
                
                st.session_state.data = temp_data
                st.success("Encoding applied successfully.")
                st.write("Updated Data Head:")
                st.write(st.session_state.data.head())
            except Exception as e:
                st.error(f"An error occurred during encoding: {str(e)}")

    # Model configuration
    st.subheader(":grey[Model Configuration]")
    target_col = st.selectbox("Select target column", options=list(st.session_state.data.columns))
    feature_col = st.multiselect(
        "Select feature columns",
        options=[col for col in st.session_state.data.columns if col != target_col]
    )

    if target_col and feature_col:
        x = st.session_state.data[feature_col]
        y = st.session_state.data[target_col]

        test_size = st.slider("Test Size (as a fraction)", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

        if problem_type == "Regression":
            model_choice = st.selectbox("Choose regression model", options=["Linear Regression", "Random Forest Regressor"])
            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor()
        else:  # Classification
            model_choice = st.selectbox("Choose classification model", options=["Logistic Regression", "Random Forest Classifier"])
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            else:
                model = RandomForestClassifier()

        if st.button("Run Model"):
            try:
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)

                st.subheader("Model Results")
                if problem_type == "Regression":
                    mse = mean_squared_error(y_test, predictions)
                    st.write(f"Mean Squared Error: {mse}")
                else:  # Classification
                    accuracy = accuracy_score(y_test, predictions)
                    st.write(f"Accuracy: {accuracy}")
                    st.text("Classification Report")
                    st.text(classification_report(y_test, predictions))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please select both a target column and feature columns to proceed.")


# Footer
st.markdown("""
"""
---
👨‍💻 Developed by Abhishek Singh
            
📧 Contact: Abhishekchhonkar2002@gmail.com
            
🔗 [LinkedIn](https://www.linkedin.com/in/itsabhisheksingh2111/) | 📂 [GitHub](https://github.com/abhishekchhonkar21/Data-analytics-app/tree/main) | 🌐 [Portfolio](https://abhishek-portfolio2023.netlify.app/)  


---
""", unsafe_allow_html=True)
