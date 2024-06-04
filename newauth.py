import boto3
cognito_client = boto3.client('cognito-idp', region_name='your-region')

response = cognito_client.initiate_auth(
    ClientId='your-app-client-id',
    AuthFlow='USER_PASSWORD_AUTH',
    AuthParameters={
        'USERNAME': 'user@example.com',
        'PASSWORD': 'yourpassword'
    }
)

print(response)
