import argparse
import json
import logging
import sys
import google.cloud.aiplatform as aip

# --- Spotify setup ---
PROJECT_ID = "de2025-471807"
REGION = "us-central1"

PIPELINE_DEF = "pipeline/build/spotify_churn_pipeline.json"
PIPELINE_ROOT = "gs://spotify_temp/pipeline_root"
PARAMETERS = "pipeline/parameters/parameters.json"


def run_pipeline_job(name, pipeline_def, pipeline_root, parameter_dict):
    # Load parameters from JSON
    with open(parameter_dict, "r") as f:
        data = json.load(f)

    logging.info("📦 Loaded parameters:")
    logging.info(json.dumps(data, indent=2))

    # Initialize Vertex AI
    aip.init(project=PROJECT_ID, location=REGION, staging_bucket="gs://spotify_temp")

    # Run pipeline job
    job = aip.PipelineJob(
        display_name=name,
        enable_caching=False,
        template_path=pipeline_def,
        pipeline_root=pipeline_root,
        parameter_values=data,
    )
    job.run()
    logging.info("🎯 Spotify churn pipeline submitted successfully.")


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="spotify-churn-lr", help="Pipeline name")
    parser.add_argument("--pipeline_def", type=str, default=PIPELINE_DEF, help="Pipeline definition file")
    parser.add_argument("--pipeline_root", type=str, default=PIPELINE_ROOT, help="GCS path for pipeline_root")
    parser.add_argument("--parameter_dict", type=str, default=PARAMETERS, help="Path to parameters JSON file")
    return vars(parser.parse_args())


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    run_pipeline_job(**parse_command_line_arguments())
