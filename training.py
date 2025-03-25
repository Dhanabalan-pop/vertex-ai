# Step 7: Define the pipeline
print("Step 7: Defining the pipeline...")
@dsl.pipeline(  # Define a KFP pipeline
    pipeline_root=f"{BUCKET_NAME}/pipeline_root",  # Set the pipeline root directory
)
def automl_iris_pipeline(  # Define the pipeline function
    project: str = PROJECT_ID,  # Input: Google Cloud project ID (default from configuration)
    location: str = REGION,  # Input: Google Cloud region (default from configuration)
    dataset_uri: str = "gs://test-bucket-sapient-dannyy/iris_updated.csv",  # Input: URI of the Iris dataset
    model_display_name: str = "automl-iris-model",  # Input: Display name for the model
    endpoint_display_name: str = "automl-iris-endpoint",  # Input: Display name for the endpoint
):
    dataset_op = gcc_aip_dataset.TabularDatasetCreateOp(  # Create a TabularDatasetCreate operation
        display_name="iris-dataset",  # Set the dataset display name
        gcs_source=dataset_uri,  # Set the GCS source URI
        project=project,  # Set the project ID
        location=location,  # Set the region
    )

    train_op = train_automl_model(  # Call the training component
        training_data=dataset_op.outputs["dataset"],  # Pass the dataset output
        project=project,  # Pass the project ID
        location=location,  # Pass the region
        display_name=model_display_name,  # Pass the model display name
    )
print("Step 7: Pipeline defined.")
