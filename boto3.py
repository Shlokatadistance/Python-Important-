import boto3
session = boto3.Session(
    aws_access_key_id='AWS_ACCESS_KEY_ID',
    aws_secret_access_key='AWS_ACCESS_SECRET'
)
a3 = session.resource('s3')
s3.meta.client.upload_file(filename = 'input_file_path',Bucket='Bucket_name',Key='s3_output_key')
