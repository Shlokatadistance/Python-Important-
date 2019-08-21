import boto3
if __name__ == "main":
    filename = '5c7775e1e7191.png'
    bucket = 'docboyz'
    client = boto3.client('rekognition')
    response = client.detect-labels(Image = {'S3Object':{'Bucket':bucket,'Name':filename}})
    print('Detected labels for',filename)
    for label in response['Labels']:
        print (label['Name'] + ' : ' + str(label['Confidence']))
