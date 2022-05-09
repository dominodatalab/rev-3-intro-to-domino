#Make sure to update line 12 with proper URL before running

import requests
import json
import os

#set up API key and headers
api_key = os.environ['DOMINO_USER_API_KEY']
headers = {'X-Domino-Api-Key': api_key,  'Content-Type': 'application/json'}  

#base url
base_url = 'https://rev.workshop.domino.tech' #i.e. 'https://mycompany.domino.tech'

#get user id
url_user = '{base_url}/v4/users/self'.format(base_url=base_url)
r_user = requests.get(url_user, headers=headers)
user_id = r_user.json()['id']

#get project id
project_name = os.environ['DOMINO_PROJECT_NAME']
url_project = '{base_url}/v4/projects?name={project_name}&ownerId={user_id}'.format(
    base_url=base_url, project_name=project_name, user_id=user_id)
r_project = requests.get(url_project, headers=headers)
project_id = r_project.json()[0]['id']

#set project tag
url_tags = base_url + '/v4/projects/{project_id}/tags'.format(project_id=project_id)
values = """
  {
    "tagNames": [
      "time-series"
    ]
  }
"""

r = requests.post(url_tags, headers=headers, data=values)