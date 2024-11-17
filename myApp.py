import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
st.set_page_config(
    page_title="Daily Use Analytics Portal",
    page_icon="🔹"
)

#title
st.title("My :red[Data] :blue[Analytics] Portal")
st.subheader(":grey[Explore data with ease]",divider="orange")
file =  st.file_uploader("Drop CSV",type=["csv","xlss"])

if(file!=None):
    if(file.name.endswith("csv")):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)
    st.dataframe(data)
    st.info("File is successfully uploaded",icon="🚨")

    st.subheader("Basic information of the dataset", divider="green")
    tab1,tab2,tab3,tab4 = st.tabs(["Summary", "Top and Bottom Rows","Data Types","Columns"])

    with tab1:
        st.subheader(":grey[Statistical summary of the dataset]")
        st.write(f'There are {data.shape[0]} rows and {data.shape[1]} columns in the dataset')
        st.dataframe(data.describe())
    with tab2:
        # st.subheader(":grey[Top rows]")
        # toprows = st.slider("Number of row you want",1,data.shape[0],key="topslider")
        # st.dataframe(data.head(toprows))
        # st.subheader(":grey[Bottom rows]")
        # bottomrows = st.slider("Number of row you want",1,data.shape[0],key="bottomslider")
        # st.dataframe(data.tail(bottomrows))

        #Top Rows
        st.subheader(":grey[Top rows]")
        toprows_input = st.text_input("Enter the number of top rows you want:", value="5", key="topinput")
        if st.button("Show Top Rows"):
            if toprows_input.isdigit():
                toprows = int(toprows_input)
                toprows = min(toprows, data.shape[0])  # Ensure it doesn't exceed the number of rows
                st.dataframe(data.head(toprows))
            else:
                st.error("Please enter a valid number.")

        # Bottom rows
        st.subheader(":grey[Bottom rows]")
        bottomrows_input = st.text_input("Enter the number of bottom rows you want:", value="5", key="bottominput")
        if st.button("Show Bottom Rows"):
            if bottomrows_input.isdigit():
                bottomrows = int(bottomrows_input)
                bottomrows = min(bottomrows, data.shape[0])  # Ensure it doesn't exceed the number of rows
                st.dataframe(data.tail(bottomrows))
            else:
                st.error("Please enter a valid number.")
    with tab3:
        st.subheader(":grey[Data types of column]")
        st.dataframe(data.dtypes)
        
    with tab4:
        st.subheader(":grey[Column names in dataset]")
        st.write(list(data.columns))
        
    st.subheader("Column values to count", divider="green")
    
    with st.expander("Value Count"):
        col1,col2 = st.columns(2)
        with col1:
            column = st.selectbox("Choose column name", options=list(data.columns))
        with col2:
            toprows = st.number_input("Top rows",min_value=1, step=1)
        count = st.button("Count")
        if (count==True):
            result = data[column].value_counts().reset_index().head(toprows)
            st.dataframe(result)

            st.subheader("Visualization",divider="grey")
            fig = px.bar(data_frame=result, x=column, y="count",text="count",template="plotly_dark")
            st.plotly_chart(fig)
            fig = px.line(data_frame=result, x=column, y="count",text="count",template="plotly_dark")
            st.plotly_chart(fig)
            fig = px.pie(data_frame=result, names=column, values="count",template="plotly_dark")
            st.plotly_chart(fig)

    st.subheader('Groupby : Simplify your data analysis',divider='green')
    st.write('The groupby lets you summarize data by specific categories and groups')
    with st.expander("Groupby"):
        col1,col2,col3 = st.columns(3)
        with col1:
            groupby_cols = st.multiselect("Choose column to groupby", options=list(data.columns))
        with col2:
            operation_col = st.selectbox("Choose column for operation", options=list(data.columns))
        with col3:
            operation = st.selectbox("Choose operation", options=["mean","median","max","min","count","sum"])
        if(groupby_cols):
            result = data.groupby(groupby_cols).agg(
            newcol = (operation_col,operation)
            ).reset_index()
            st.dataframe(result)
            
            st.subheader(":grey[Data Visualization]",divider="grey")
            graph = st.selectbox("Choose your graphs",options=['line','bar','scatter','pie','sunburst'])
            if(graph=='line') :
                x_axis = st.selectbox('Choose X axis',options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis',options=list(result.columns))
                color = st.selectbox('Color Information',options=[None]+list(result.columns))
                fig = px.line(data_frame=result, x=x_axis, y=y_axis, color=color,markers="o")
                st.plotly_chart(fig)
            
            elif(graph=='bar'):
                x_axis = st.selectbox('Choose X axis',options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis',options=list(result.columns))
                color = st.selectbox('Color Information',options=[None]+list(result.columns))
                facet_col = st.selectbox("Column Information",options=[None]+list(result.columns))
                fig = px.bar(data_frame=result, x=x_axis, y=y_axis, color=color,facet_col=facet_col,barmode="group")
                st.plotly_chart(fig)

            elif(graph=='scatter'):
                x_axis = st.selectbox('Choose X axis',options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis',options=list(result.columns))
                color = st.selectbox('Color Information',options=[None]+list(result.columns))
                size = st.selectbox("Size Column",options=[None]+list(result.columns))
                fig = px.scatter(data_frame=result, x=x_axis, y=y_axis, color=color,size=size)
                st.plotly_chart(fig)

            elif graph == 'pie':
                st.subheader("Pie Chart")
                names = st.selectbox('Choose Labels', options=list(result.columns))
                values = st.selectbox('Choose Numerical Values', options=list(result.columns))
                fig = px.pie(data_frame=result, values=values, names=names)
                st.plotly_chart(fig)

            elif graph == 'sunburst':
                st.subheader("Sunburst Chart")
                path = st.multiselect('Choose Your Path (Hierarchy)', options=list(result.columns))
                values = st.selectbox('Choose Numerical Values for Sunburst', options=list(result.columns))
                fig = px.sunburst(data_frame=result, path=path, values=values)
                st.plotly_chart(fig)
               

    st.header(":blue[Apply Machine Learning]", divider="green")
    st.subheader(":grey[Choose Problem Type]")

    # Choose problem type
    problem_type = st.radio("Select One Machine Learning Model", ["Regression", "Classification"])

    # Preprocessing: Option to ignore rows with null values
    st.subheader(":grey[Preprocessing]")
    ignore_nulls = st.checkbox("Ignore rows with null values during training")
    if ignore_nulls:
        data = data.dropna()
        st.info("Rows with null values have been removed.", icon="✅")
    else:
        st.warning("Rows with null values remain in the dataset. Ensure your model can handle them.", icon="⚠️")

    # Model configuration
    st.subheader(":grey[Model Configuration]")
    target_col = st.selectbox("Select target column", options=list(data.columns))
    feature_col = st.multiselect(
    "Select feature columns", options=[col for col in data.columns if col != target_col]
    )

    if target_col and feature_col:
        x = data[feature_col]
        y = data[target_col]

    # Split data
        test_size = st.slider("Test Size (as a fraction)", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

        # Select model based on problem type
        if problem_type == "Regression":
            model_choice = st.selectbox("Choose regression model", options=["Linear Regression", "Random Forest Regressor"])
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest Regressor":
                model = RandomForestRegressor()

        elif problem_type == "Classification":
            model_choice = st.selectbox("Choose classification model", options=["Logistic Regression", "Random Forest Classifier"])
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Random Forest Classifier":
                model = RandomForestClassifier()

        # Run the model
        if st.button("Run Model"):
            try:
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)

                st.subheader("Model Results")
                if problem_type == "Regression":
                    mse = mean_squared_error(y_test, predictions)
                    st.write(f"Mean Squared Error: {mse}")
                elif problem_type == "Classification":
                    accuracy = accuracy_score(y_test, predictions)
                    st.write(f"Accuracy: {accuracy}")
                    st.text("Classification Report")
                    st.text(classification_report(y_test, predictions))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please select both a target column and feature columns to proceed.")
