# Step 5: Define the AutoML training component
print("Step 5: Defining AutoML training component...")
@component(  # Define a KFP component
    base_image="python:3.9",  # Use Python 3.9 as the base image
    packages_to_install=["google-cloud-aiplatform"],  # Install Vertex AI library in the component
)
def train_automl_model(  # Define the training component function
    training_data: Input[Artifact],  # Input: Training dataset
    model: Output[Model],  # Output: Trained model
    project: str,  # Input: Google Cloud project ID
    location: str,  # Input: Google Cloud region
    display_name: str,  # Input: Display name for the model
):
    import subprocess
    subprocess.run(["pip", "install", "click>=8.0.0,<9", "kfp-pipeline-spec==0.5.0", "kfp-server-api>=2.1.0,<2.5.0", "kubernetes>=8.0.0,<31", "PyYAML>=5.3,<7", "requests-toolbelt>=0.8.0,<2", "tabulate>=0.8.6,<1","protobuf<5,>=4.21.1","urllib3<2.0.0","protobuf<6.0dev,>=5.26.1"])
    from google.cloud import aiplatform  # Import Vertex AI library inside the component

    aiplatform.init(project=project, location=location)  # Initialize Vertex AI SDK inside the component
    job = aiplatform.AutoMLTabularTrainingJob(  # Create an AutoML Tabular Training Job
        display_name=display_name,  # Set the display name
        optimization_prediction_type="classification",  # Set the prediction type to classification
    )
    my_model = job.run(  # Run the training job
        dataset=training_data.uri,  # Use the training dataset URI
        target_column="label",  # Specify the target column
        model_display_name=display_name,  # Set the model display name
        training_fraction_split=0.8,  # Set training split fraction
        validation_fraction_split=0.1,  # Set validation split fraction
        test_fraction_split=0.1,  # Set test split fraction
    )
    model.uri = my_model.resource_name  # Set the output model URI
print("Step 5: AutoML training component defined.")
