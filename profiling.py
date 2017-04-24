from pymongo import MongoClient
from insertIntoDb import insert
from getTweets import lookupTweets

client = MongoClient('localhost:27017', serverSelectionTimeoutMS=1000)

db = client['user-details']
collection = db['status']

if db.collection.count() < lookupTweets():
	db.collection.remove()
	insert()
else:
	print("Hello world", lookupTweets())