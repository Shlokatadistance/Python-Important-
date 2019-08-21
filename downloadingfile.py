"""
import boto3
s3 = boto3.client('s3')
s3.download_file('dockboyz','1563775879837-9325.docboyz.jpg','s3downloads.jpg')
s3 = boto3.client('s3')
with open('FILE_NAME','wb') as f:
    s3.download_fileobj('dockboyz','1563775879837-9325.docboyz.jpg',f)
 
import boto3
s3 = boto3.client('s3')
filename = 'file.txt'
bucket_name = 'my-bucket'
s3.upload_file(filename,bucket_name,filename)
"""
import boto3
import botocore

BUCKET_NAME = 'dockboyz' # replace with your bucket name
KEY = 'uploads/pickups/1563775879837-9325.docboyz.jpg' # replace with your object key

s3 = boto3.resource('s3')

try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, 'my_local_image.jpg')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise
#the credentials for the amazon thing is SHLOK and the region i gave in is MUMABI,
    #the program is reading in those credentials and using that region, so i need to give in the correct
    #region name
#its either sudo apt-get or sudo yum
