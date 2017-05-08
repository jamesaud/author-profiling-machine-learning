NOTES:
Requires python 3.5
Requires pymongo: 
    'pip install pymongo'
Run from docker (this starts mongo db and optionally the python container)  with 
    'docker-compose up'

You can run inside the python container with: 
    "docker exec test_python_1 python mongo.py"

Or enter the container with: 
    "docker exec -it test_python_1 /bin/bash" and run "python mongo.py"

Hint: see running containers with 'docker ps'
