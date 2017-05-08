
from pymongo import MongoClient
from pprint import pprint

### CLIENT SETUP ###
# If running inside python Docker container
client = MongoClient('mongo:27017', serverSelectionTimeoutMS=1000) 
# Else if running locally
client = MongoClient('localhost:27017', serverSelectionTimeoutMS=1000)


### CODE ###
db = client.test

# Insert test record 
result = db.restaurants.insert_one(
    {
        "test": "myobject",
        "restaurant_id": "41704620",
        "category": 'FEMALE-18-24'
    }
)

pprint(db.restaurants.find_one())
