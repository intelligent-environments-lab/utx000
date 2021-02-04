# Import two classes from the boxsdk module - Client and OAuth2
from boxsdk import Client, OAuth2
from boxsdk.network.default_network import DefaultNetwork

# Operation
from pprint import pformat

# Define client ID, client secret, and developer token.
CLIENT_ID = None
CLIENT_SECRET = None
ACCESS_TOKEN = None

# Read app info from text file
with open('../../admin/box_app.txt', 'r') as app_cfg:
    CLIENT_ID = app_cfg.readline()
    CLIENT_SECRET = app_cfg.readline()
    ACCESS_TOKEN = app_cfg.readline()

# Create OAuth2 object. It's already authenticated, thanks to the developer token.
auth = OAuth2(CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN)

# Create the authenticated client
client = Client(auth)

root_folder = client.root_folder().get()

items = root_folder.get_items()
for item in items:
    print('{0} {1} is named "{2}"'.format(item.type.capitalize(), item.id, item.name))