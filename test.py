import boto3
import json

# Configuration
endpoint_name = "jumpstart-dft-meta-textgeneration-l-20260221-092648"
client = boto3.client("runtime.sagemaker", region_name="us-east-1")

# Payload for Meta Llama/JumpStart models
payload = {
    "inputs": "What is the capital of France?",
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }
}

# Invoke endpoint
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload)
)

# Parse response
result = json.loads(response["Body"].read().decode())
print(result)