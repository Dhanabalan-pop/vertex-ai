from kfp import components
from kfp.dsl import Input, Output, Artifact, Model
import logging
from typing import NamedTuple

@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform==1.48.0",
        "protobuf==4.21.6",
        "kfp==2.10.1",
        "kfp-pipeline-spec==0.5.0",
        "kfp-server-api==2.3.0",
        "urllib3==1.26.16"
    ]
)
def train_automl_model(
    training_data: Input[Artifact],
    model: Output[Model],
    project: str,
    location: str,
    display_name: str,
    target_column: str = "label",
    prediction_type: str = "classification",
    training_split: float = 0.8,
    validation_split: float = 0.1,
    test_split: float = 0.1
) -> NamedTuple('Outputs', [
    ('model_name', str),
    ('training_job_id', str)
]):
    """Trains an AutoML Tabular model with comprehensive logging.
    
    Args:
        training_data: Input dataset artifact
        model: Output model artifact
        project: GCP project ID
        location: GCP region
        display_name: Display name for the model
        target_column: Name of the target column
        prediction_type: Type of prediction ('classification' or 'regression')
        training_split: Fraction of data for training (0-1)
        validation_split: Fraction of data for validation (0-1)
        test_split: Fraction of data for testing (0-1)
        
    Returns:
        NamedTuple containing:
            - model_name: Resource name of the trained model
            - training_job_id: ID of the training job
    """
    from google.cloud import aiplatform
    from google.api_core.exceptions import GoogleAPICallError
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting AutoML training component")
    
    try:
        # Initialize Vertex AI
        logger.info(f"Initializing Vertex AI for project: {project}, location: {location}")
        aiplatform.init(project=project, location=location)
        
        # Create AutoML training job
        logger.info(f"Creating AutoML training job with display name: {display_name}")
        logger.info(f"Prediction type: {prediction_type}")
        
        job = aiplatform.AutoMLTabularTrainingJob(
            display_name=display_name,
            optimization_prediction_type=prediction_type
        )
        
        # Run training job
        logger.info(f"Starting training job with dataset: {training_data.uri}")
        logger.info(f"Data splits - Train: {training_split}, Val: {validation_split}, Test: {test_split}")
        
        my_model = job.run(
            dataset=training_data.uri,
            target_column=target_column,
            model_display_name=display_name,
            training_fraction_split=training_split,
            validation_fraction_split=validation_split,
            test_fraction_split=test_split,
            sync=True  # Wait for job completion
        )
        
        # Set output artifacts
        model.uri = my_model.resource_name
        logger.info(f"Training completed successfully. Model URI: {model.uri}")
        logger.info(f"Model resource name: {my_model.resource_name}")
        
        # Return additional information
        from collections import namedtuple
        output = namedtuple('Outputs', ['model_name', 'training_job_id'])
        return output(
            model_name=my_model.resource_name,
            training_job_id=job._gca_resource.name
        )
        
    except GoogleAPICallError as e:
        logger.error(f"Google API call failed: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Invalid parameter value: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise
