import boto3

client = boto3.client('s3', region_name='ap-south-1')

client.upload_file('Videos/download.jpg', 'dockboyz', 'download.jpg')
"""
IAM role lets resources interact between accounts

