import os, re
import xml.etree.ElementTree as ET

parser = ET.XMLParser(encoding="utf-8")
file = open('raw-dataset/truth.txt','r')
fileList = file.readlines()
file.close()

consumer_key = 'ompV1RA39zhlejhRYrYi7y6F3'
consumer_secret = '5XkJD67sa42ys3KMNKpn6wwtvwCNgYviFdOfo9IQifd0rIV4w8'
access_token = '144817473-feu73Ni9oZJ55O4H2CP7liWKJUB959zTM3Cc5lUu'
access_secret = 'Tr7IsVOi6dd5RTGNeAYGHAFEmiJcrOhznN3anKJM9aj64'

if "processed-data" in os.listdir():
	os.chdir("processed-data")
else:
	os.mkdir("processed-data")
	os.chdir("processed-data")

def tweet_text_by_id(id, consumer_key=None, consumer_secret=None, access_token=None, access_token_secret=None):
    import tweepy
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    tweet = api.get_status(id)
    return tweet.text

errorFile = open('errorlog.txt','a')
statusCnt = 0
for file in fileList:
	tempFileName = file.strip().split(":::")
	file = open('../raw-dataset/'+tempFileName[0]+'.xml','r',encoding='utf-8')
	content = file.read()
	file.close()
	statusIds = re.findall('(?<=<document id=").+?(?=")',content,re.DOTALL)
	
	try:
		os.mkdir(tempFileName[1]+"-"+tempFileName[2])
	except:
		print(tempFileName[1]+"-"+tempFileName[2]+" exists")
	os.chdir(tempFileName[1]+"-"+tempFileName[2])
	for status in statusIds:
		statusCnt +=1
		if status+'.txt' not in os.listdir():
			file = open(status+'.txt','w',encoding='utf-8')
			try:
				file.write(tweet_text_by_id(status, consumer_key, consumer_secret, access_token, access_secret))
			except Exception as e:
				errorFile.write(str(e))
			else:
				file.close()
				print(statusCnt)
		else:
			print(statusCnt)
	os.chdir("../")
errorFile.close()
