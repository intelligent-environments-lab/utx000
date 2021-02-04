# Import two classes from the boxsdk module - Client and OAuth2
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import webbrowser
# Operation
from pprint import pformat

from boxsdk import Client, OAuth2
from boxsdk.network.default_network import DefaultNetwork

#AUTH_CODE_URI = None

# Read app info from text file
with open('../../admin/box_app.txt', 'r') as app_cfg:
    CLIENT_ID = app_cfg.readline()
    CLIENT_SECRET = app_cfg.readline()
    ACCESS_TOKEN = app_cfg.readline()

def store_tokens(access_token, refresh_token):
    tokens = {'access_token':access_token,'refresh_token':refresh_token}

    with open('../../admin/box_app.txt','w') as file:
        json.dumps(tokens, file)

oauth = OAuth2(
    client_id='h315541mc5229opkgyjacd3j71kxuaq7',
    client_secret='7OqJAYCX9oXnvuO5VBlViWbMhHJMGyBj',
    store_tokens=store_tokens,
)

auth_url, csrf_token = oauth.get_authorization_url('http://localhost:8000')


def run(auth_url):
    webbrowser.open(auth_url)
    class SrvHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            run.REDIRECT_URI = self.path

            self.send_response(303)
            self.send_header('Location','http://box.com')
            self.end_headers()

            raise KeyboardInterrupt

    try:
        server_class=HTTPServer
        server_address = ('', 8000)
        handler_class=SrvHandler
        httpd = server_class(server_address, handler_class)
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    
    redirect_url = run.REDIRECT_URI
    return redirect_url

print(run(auth_url))

# Create OAuth2 object. It's already authenticated, thanks to the developer token.
auth = OAuth2(CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN)

# Create the authenticated client
client = Client(auth)

root_folder = client.root_folder().get()

items = root_folder.get_items()
for item in items:
    print('{0} {1} is named "{2}"'.format(item.type.capitalize(), item.id, item.name))
