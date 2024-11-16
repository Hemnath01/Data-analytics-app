import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import time

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
                values = st.selectbox('Choose Numerical Values', options=list(result.columns))
                names = st.selectbox('Choose Labels', options=list(result.columns))
                fig = px.pie(data_frame=result, values=values, names=names)
                st.plotly_chart(fig)

            elif graph == 'sunburst':
                st.subheader("Sunburst Chart")
                path = st.multiselect('Choose Your Path (Hierarchy)', options=list(result.columns))
                values = st.selectbox('Choose Numerical Values for Sunburst', options=list(result.columns))
                fig = px.sunburst(data_frame=result, path=path, values=values)
                st.plotly_chart(fig)
               
