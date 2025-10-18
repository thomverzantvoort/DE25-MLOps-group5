import argparse
from google.cloud import aiplatform

def run_pipeline(name, pipeline_def, pipeline_root, parameter_dict):
    print(f"Running pipeline: {name}")
    aiplatform.init(project=aiplatform.gapic.utils.get_project(), location=aiplatform.gapic.utils.get_location())
    job = aiplatform.PipelineJob(
        display_name=name,
        template_path=pipeline_def,
        pipeline_root=pipeline_root,
        parameter_values=json.load(open(parameter_dict))
    )
    job.run(sync=True)

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--pipeline_def")
    parser.add_argument("--pipeline_root")
    parser.add_argument("--parameter_dict")
    args = parser.parse_args()
    run_pipeline(args.name, args.pipeline_def, args.pipeline_root, args.parameter_dict)
