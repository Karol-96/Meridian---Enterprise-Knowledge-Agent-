import boto3
import os
from dotenv import load_dotenv
import json



load_dotenv()

client = boto3.client("bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)



# Test 1 - Titan Embed
response = client.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({"inputText": "hello meridian"})
)
result = json.loads(response["body"].read())
print("Titan Embed works! Dims:", len(result["embedding"]))

# Test 2 - Claude
response = client.converse(
modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": [{"text": "say hello in one sentence"}]}]
)
print("Claude works!", response["output"]["message"]["content"][0]["text"])

