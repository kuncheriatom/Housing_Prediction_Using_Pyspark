import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col, when

# Initialize Spark session
spark = SparkSession.builder.appName("LoanPrediction").getOrCreate()

@st.cache_resource
def load_rf_model():
    # Load the RandomForest model from the specified path
    return RandomForestClassificationModel.load(r"rf_model")

@st.cache_resource
def get_vector_assembler():
    # Define the VectorAssembler with the appropriate input and output column names
    return VectorAssembler(
        inputCols=["BASEMENTAREA_MODE", "NONLIVINGAREA_AVG", "APARTMENTS_MODE",
                   "LIVE_REGION_NOT_WORK_REGION", "CODE_GENDER_index",
                   "REGION_POPULATION_RELATIVE", "FLAG_DOCUMENT_17", "ELEVATORS_AVG",
                   "REQUESTED_GOODS_PRICE", "EXTERNAL_CREDIT_SCORE"],
        outputCol="features"
    )

def predict(input_data, rf_model, vector_assembler):
    # Convert input data to Spark DataFrame
    input_df = spark.createDataFrame([input_data])
    
    # Get the VectorAssembler and transform the data to create feature vectors
    assembled_data = vector_assembler.transform(input_df)
    
    # Rename the 'features' column to 'top_features' to match the model's expectation
    final_data = assembled_data.withColumnRenamed("features", "top_features")
    
    # Apply the RandomForest model to make predictions
    predictions = rf_model.transform(final_data)
    
    # Add an interpretation column based on the prediction
    predictions_interpreted = predictions.withColumn(
        "default_interpretation",
        when(col("prediction") >= 0.5, "Likely to Default").otherwise("Not Likely to Default")
    )
    
    # Collect the result
    result = predictions_interpreted.select("default_interpretation").collect()[0][0]
    return result

def main():
    st.title("Loan Prediction Model")

    # Load the RandomForest model and VectorAssembler
    rf_model = load_rf_model()
    vector_assembler = get_vector_assembler()
    
    # Define input fields with better descriptions
    st.header("Input Loan Application Data")
    
    BASEMENTAREA_MODE = st.number_input("Basement Area (in square meters)", value=10.0)
    NONLIVINGAREA_AVG = st.number_input("Average Non-Living Area (in square meters)", value=60.0)
    APARTMENTS_MODE = st.number_input("Number of Apartments", value=3.0)
    LIVE_REGION_NOT_WORK_REGION = st.selectbox("Region Type", [0.0, 1.0], format_func=lambda x: "Working Area" if x == 0.0 else "Non-Working Area")
    CODE_GENDER_index = st.selectbox("Gender", [0.0, 1.0], format_func=lambda x: "Male" if x == 0.0 else "Female")
    REGION_POPULATION_RELATIVE = st.number_input("Relative Population of Region", value=0.02)
    FLAG_DOCUMENT_17 = st.selectbox("Document 17 Missing", [0.0, 1.0], format_func=lambda x: "No" if x == 0.0 else "Yes")
    ELEVATORS_AVG = st.number_input("Average Number of Elevators", value=2.0)
    REQUESTED_GOODS_PRICE = st.number_input("Requested Loan Price (in currency)", value=500000.0)
    EXTERNAL_CREDIT_SCORE = st.number_input("External Credit Score", value=0.7)
    
    # Collect input data
    input_data = {
        "BASEMENTAREA_MODE": float(BASEMENTAREA_MODE),
        "NONLIVINGAREA_AVG": float(NONLIVINGAREA_AVG),
        "APARTMENTS_MODE": float(APARTMENTS_MODE),
        "LIVE_REGION_NOT_WORK_REGION": float(LIVE_REGION_NOT_WORK_REGION),
        "CODE_GENDER_index": float(CODE_GENDER_index),
        "REGION_POPULATION_RELATIVE": float(REGION_POPULATION_RELATIVE),
        "FLAG_DOCUMENT_17": float(FLAG_DOCUMENT_17),
        "ELEVATORS_AVG": float(ELEVATORS_AVG),
        "REQUESTED_GOODS_PRICE": float(REQUESTED_GOODS_PRICE),
        "EXTERNAL_CREDIT_SCORE": float(EXTERNAL_CREDIT_SCORE)
    }

    if st.button("Predict"):
        # Make predictions
        prediction = predict(input_data, rf_model, vector_assembler)
        
        # Display prediction result
        st.write(f"Predicted Loan Status: {prediction}")

if __name__ == "__main__":
    main()
