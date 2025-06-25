import boto3
import pickle

s3 = boto3.client('s3')

# Download and load a pickled file
response = s3.get_object(Bucket='staging-data-ingestion-bucket', Key='analytics/clean_trades')
data = pickle.loads(response['Body'].read())
print(data)  # This should show your DataFrame