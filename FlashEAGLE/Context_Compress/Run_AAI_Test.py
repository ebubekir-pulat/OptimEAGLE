# Reference for below code: https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html#submitting-a-ray-job

from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("FILL_IN")
job_id = client.submit_job(
    entrypoint="python AAI_Test.py True True",
    runtime_env={"working_dir": "./", "pip":["datasets==3.6.0", "nltk"]},
    entrypoint_num_gpus=1,
    entrypoint_num_cpus=12,
    submission_id="OptimEAGLE-AAI-Test-Official-EAGLE3-Translate-5"
)
print(job_id)