import os
from googleapiclient.discovery import build
import pandas as pd
from IPython.display import JSON
import urllib.request
import re
import re
import pickle
import string
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import nltk
# nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize

import keras
from keras.preprocessing import sequence
stopword = set(stopwords.words('english'))

import warnings
warnings.filterwarnings("ignore")




class LiveYouTubeComments:

    def __init__(self):
            # Obtiene la ruta absoluta del archivo actual
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, "../Model")
            data_dir = os.path.join(current_dir, "../Data/LiveYT/liveComments.csv")
            data_dir = os.path.join(current_dir, "../Data/LiveYT")  # Change to the directory containing the CSV file
            self.data_file = os.path.join(data_dir, "liveComments.csv")

            self.youtube = build(
                "youtube", "v3", developerKey="XXXX")
            self.load_model = keras.models.load_model(os.path.join(model_dir, "hate_model.h5"))
            with open(os.path.join(model_dir, "tokenizer.pickle"), 'rb') as handle:
                self.load_tokenizer = pickle.load(handle)


    def get_comments_in_live_videos(self, video_ids, limit):
        all_comments = []

        for video_id in video_ids:
            try:
                # Get the live chat ID for the video
                live_chat_id = self.youtube.videos().list(
                    part="liveStreamingDetails",
                    id=video_id
                ).execute()['items'][0]['liveStreamingDetails']['activeLiveChatId']

                # Get the live chat messages for the video
                request = self.youtube.liveChatMessages().list(
                    liveChatId=live_chat_id,
                    part="snippet",
                    maxResults=limit
                )
                response = request.execute()

                comments_in_video = [
                    {
                        'comment': comment['snippet']['textMessageDetails']['messageText'],
                        'date': comment['snippet']['publishedAt'],
                        'user': comment['snippet']['authorChannelId']
                    }
                    for comment in response['items']
                ]

                for comment_info in comments_in_video:
                    comment_info['video_id'] = video_id
                    all_comments.append(comment_info)

            except Exception as e:
                print(f"Could not get comments for video {video_id}: {str(e)}")

        return pd.DataFrame(all_comments)

    def extract_videos_ids(self, urls):
        all_videos = []
        for url in urls:
            pattern = r'.*=(.+)'
            matches = re.findall(pattern, url)
            all_videos.append(matches[0])

        return all_videos

    def clean_text(self, text): 
        text = str(text).lower()

        # Special characters
        text = re.sub(r"\x89Û_", "", text)
        text = re.sub(r"\x89ÛÒ", "", text)
        text = re.sub(r"\x89ÛÓ", "", text)
        text = re.sub(r"\x89ÛÏWhen", "When", text)
        text = re.sub(r"\x89ÛÏ", "", text)
        text = re.sub(r"China\x89Ûªs", "China's", text)
        text = re.sub(r"let\x89Ûªs", "let's", text)
        text = re.sub(r"\x89Û÷", "", text)
        text = re.sub(r"\x89Ûª", "", text)
        text = re.sub(r"\x89Û\x9d", "", text)
        text = re.sub(r"å_", "", text)
        text = re.sub(r"\x89Û¢", "", text)
        text = re.sub(r"\x89Û¢åÊ", "", text)
        text = re.sub(r"fromåÊwounds", "from wounds", text)
        text = re.sub(r"åÊ", "", text)
        text = re.sub(r"åÈ", "", text)
        text = re.sub(r"JapÌ_n", "Japan", text)    
        text = re.sub(r"Ì©", "e", text)
        text = re.sub(r"å¨", "", text)
        text = re.sub(r"SuruÌ¤", "Suruc", text)
        text = re.sub(r"åÇ", "", text)
        text = re.sub(r"å£3million", "3 million", text)
        text = re.sub(r"åÀ", "", text)
        
        # Contractions
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"there's", "there is", text)
        text = re.sub(r"We're", "We are", text)
        text = re.sub(r"That's", "That is", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"they're", "they are", text)
        text = re.sub(r"Can't", "Cannot", text)
        text = re.sub(r"wasn't", "was not", text)
        text = re.sub(r"don\x89Ûªt", "do not", text)
        text = re.sub(r"aren't", "are not", text)
        text = re.sub(r"isn't", "is not", text)
        text = re.sub(r"What's", "What is", text)
        text = re.sub(r"haven't", "have not", text)
        text = re.sub(r"hasn't", "has not", text)
        text = re.sub(r"There's", "There is", text)
        text = re.sub(r"He's", "He is", text)
        text = re.sub(r"It's", "It is", text)
        text = re.sub(r"You're", "You are", text)
        text = re.sub(r"I'M", "I am", text)
        text = re.sub(r"shouldn't", "should not", text)
        text = re.sub(r"wouldn't", "would not", text)
        text = re.sub(r"i'm", "I am", text)
        text = re.sub(r"I\x89Ûªm", "I am", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r"Isn't", "is not", text)
        text = re.sub(r"Here's", "Here is", text)
        text = re.sub(r"you've", "you have", text)
        text = re.sub(r"you\x89Ûªve", "you have", text)
        text = re.sub(r"we're", "we are", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"we've", "we have", text)
        text = re.sub(r"it\x89Ûªs", "it is", text)
        text = re.sub(r"doesn\x89Ûªt", "does not", text)
        text = re.sub(r"It\x89Ûªs", "It is", text)
        text = re.sub(r"Here\x89Ûªs", "Here is", text)
        text = re.sub(r"who's", "who is", text)
        text = re.sub(r"I\x89Ûªve", "I have", text)
        text = re.sub(r"y'all", "you all", text)
        text = re.sub(r"can\x89Ûªt", "cannot", text)
        text = re.sub(r"would've", "would have", text)
        text = re.sub(r"it'll", "it will", text)
        text = re.sub(r"we'll", "we will", text)
        text = re.sub(r"wouldn\x89Ûªt", "would not", text)
        text = re.sub(r"We've", "We have", text)
        text = re.sub(r"he'll", "he will", text)
        text = re.sub(r"Y'all", "You all", text)
        text = re.sub(r"Weren't", "Were not", text)
        text = re.sub(r"Didn't", "Did not", text)
        text = re.sub(r"they'll", "they will", text)
        text = re.sub(r"they'd", "they would", text)
        text = re.sub(r"DON'T", "DO NOT", text)
        text = re.sub(r"That\x89Ûªs", "That is", text)
        text = re.sub(r"they've", "they have", text)
        text = re.sub(r"i'd", "I would", text)
        text = re.sub(r"should've", "should have", text)
        text = re.sub(r"You\x89Ûªre", "You are", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"Don\x89Ûªt", "Do not", text)
        text = re.sub(r"we'd", "we would", text)
        text = re.sub(r"i'll", "I will", text)
        text = re.sub(r"weren't", "were not", text)
        text = re.sub(r"They're", "They are", text)
        text = re.sub(r"Can\x89Ûªt", "Cannot", text)
        text = re.sub(r"you\x89Ûªll", "you will", text)
        text = re.sub(r"I\x89Ûªd", "I would", text)
        text = re.sub(r"let's", "let us", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"you're", "you are", text)
        text = re.sub(r"i've", "I have", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"i'll", "I will", text)
        text = re.sub(r"doesn't", "does not", text)
        text = re.sub(r"i'd", "I would", text)
        text = re.sub(r"didn't", "did not", text)
        text = re.sub(r"ain't", "am not", text)
        text = re.sub(r"you'll", "you will", text)
        text = re.sub(r"I've", "I have", text)
        text = re.sub(r"Don't", "do not", text)
        text = re.sub(r"I'll", "I will", text)
        text = re.sub(r"I'd", "I would", text)
        text = re.sub(r"Let's", "Let us", text)
        text = re.sub(r"you'd", "You would", text)
        text = re.sub(r"It's", "It is", text)
        text = re.sub(r"Ain't", "am not", text)
        text = re.sub(r"Haven't", "Have not", text)
        text = re.sub(r"Could've", "Could have", text)
        text = re.sub(r"youve", "you have", text)  
        text = re.sub(r"donå«t", "do not", text)   
                
        # Character entity references
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&amp;", "&", text)
        
        # Typos, slang and informal abbreviations
        text = re.sub(r"w/e", "whatever", text)
        text = re.sub(r"w/", "with", text)
        text = re.sub(r"USAgov", "USA government", text)
        text = re.sub(r"recentlu", "recently", text)
        text = re.sub(r"Ph0tos", "Photos", text)
        text = re.sub(r"amirite", "am I right", text)
        text = re.sub(r"exp0sed", "exposed", text)
        text = re.sub(r"<3", "love", text)
        text = re.sub(r"amageddon", "armageddon", text)
        text = re.sub(r"Trfc", "Traffic", text)
        text = re.sub(r"8/5/2015", "2015-08-05", text)
        text = re.sub(r"WindStorm", "Wind Storm", text)
        text = re.sub(r"8/6/2015", "2015-08-06", text)
        text = re.sub(r"10:38PM", "10:38 PM", text)
        text = re.sub(r"10:30pm", "10:30 PM", text)
        text = re.sub(r"16yr", "16 year", text)
        text = re.sub(r"lmao", "laughing my ass off", text)   
        text = re.sub(r"TRAUMATISED", "traumatized", text)
        
        # Hashtags and usernames
        text = re.sub(r"IranDeal", "Iran Deal", text)
        text = re.sub(r"ArianaGrande", "Ariana Grande", text)
        text = re.sub(r"camilacabello97", "camila cabello", text) 
        text = re.sub(r"RondaRousey", "Ronda Rousey", text)     
        text = re.sub(r"MTVHottest", "MTV Hottest", text)
        text = re.sub(r"TrapMusic", "Trap Music", text)
        text = re.sub(r"ProphetMuhammad", "Prophet Muhammad", text)
        text = re.sub(r"PantherAttack", "Panther Attack", text)
        text = re.sub(r"StrategicPatience", "Strategic Patience", text)
        text = re.sub(r"socialnews", "social news", text)
        text = re.sub(r"NASAHurricane", "NASA Hurricane", text)
        text = re.sub(r"onlinecommunities", "online communities", text)
        text = re.sub(r"humanconsumption", "human consumption", text)
        text = re.sub(r"Typhoon-Devastated", "Typhoon Devastated", text)
        text = re.sub(r"Meat-Loving", "Meat Loving", text)
        text = re.sub(r"facialabuse", "facial abuse", text)
        text = re.sub(r"LakeCounty", "Lake County", text)
        text = re.sub(r"BeingAuthor", "Being Author", text)
        text = re.sub(r"withheavenly", "with heavenly", text)
        text = re.sub(r"thankU", "thank you", text)
        text = re.sub(r"iTunesMusic", "iTunes Music", text)
        text = re.sub(r"OffensiveContent", "Offensive Content", text)
        text = re.sub(r"WorstSummerJob", "Worst Summer Job", text)
        text = re.sub(r"HarryBeCareful", "Harry Be Careful", text)
        text = re.sub(r"NASASolarSystem", "NASA Solar System", text)
        text = re.sub(r"animalrescue", "animal rescue", text)
        text = re.sub(r"KurtSchlichter", "Kurt Schlichter", text)
        text = re.sub(r"aRmageddon", "armageddon", text)
        text = re.sub(r"Throwingknifes", "Throwing knives", text)
        text = re.sub(r"GodsLove", "God's Love", text)
        text = re.sub(r"bookboost", "book boost", text)
        text = re.sub(r"ibooklove", "I book love", text)
        text = re.sub(r"NestleIndia", "Nestle India", text)
        text = re.sub(r"realDonaldTrump", "Donald Trump", text)
        text = re.sub(r"DavidVonderhaar", "David Vonderhaar", text)
        text = re.sub(r"CecilTheLion", "Cecil The Lion", text)
        text = re.sub(r"weathernetwork", "weather network", text)
        text = re.sub(r"withBioterrorism&use", "with Bioterrorism & use", text)
        text = re.sub(r"Hostage&2", "Hostage & 2", text)
        text = re.sub(r"GOPDebate", "GOP Debate", text)
        text = re.sub(r"RickPerry", "Rick Perry", text)
        text = re.sub(r"frontpage", "front page", text)
        text = re.sub(r"NewsIntexts", "News In texts", text)
        text = re.sub(r"ViralSpell", "Viral Spell", text)
        text = re.sub(r"til_now", "until now", text)
        text = re.sub(r"volcanoinRussia", "volcano in Russia", text)
        text = re.sub(r"ZippedNews", "Zipped News", text)
        text = re.sub(r"MicheleBachman", "Michele Bachman", text)
        text = re.sub(r"53inch", "53 inch", text)
        text = re.sub(r"KerrickTrial", "Kerrick Trial", text)
        text = re.sub(r"abstorm", "Alberta Storm", text)
        text = re.sub(r"Beyhive", "Beyonce hive", text)
        text = re.sub(r"IDFire", "Idaho Fire", text)
        text = re.sub(r"DETECTADO", "Detected", text)
        text = re.sub(r"RockyFire", "Rocky Fire", text)
        text = re.sub(r"Listen/Buy", "Listen / Buy", text)
        text = re.sub(r"NickCannon", "Nick Cannon", text)
        text = re.sub(r"FaroeIslands", "Faroe Islands", text)
        text = re.sub(r"yycstorm", "Calgary Storm", text)
        text = re.sub(r"IDPs:", "Internally Displaced People :", text)
        text = re.sub(r"ArtistsUnited", "Artists United", text)
        text = re.sub(r"ClaytonBryant", "Clayton Bryant", text)
        text = re.sub(r"jimmyfallon", "jimmy fallon", text)
        text = re.sub(r"justinbieber", "justin bieber", text)  
        text = re.sub(r"UTC2015", "UTC 2015", text)
        text = re.sub(r"Time2015", "Time 2015", text)
        text = re.sub(r"djicemoon", "dj icemoon", text)
        text = re.sub(r"LivingSafely", "Living Safely", text)
        text = re.sub(r"FIFA16", "Fifa 2016", text)
        text = re.sub(r"thisiswhywecanthavenicethings", "this is why we cannot have nice things", text)
        text = re.sub(r"bbcnews", "bbc news", text)
        text = re.sub(r"UndergroundRailraod", "Underground Railraod", text)
        text = re.sub(r"c4news", "c4 news", text)
        text = re.sub(r"OBLITERATION", "obliteration", text)
        text = re.sub(r"MUDSLIDE", "mudslide", text)
        text = re.sub(r"NoSurrender", "No Surrender", text)
        text = re.sub(r"NotExplained", "Not Explained", text)
        text = re.sub(r"greatbritishbakeoff", "great british bake off", text)
        text = re.sub(r"LondonFire", "London Fire", text)
        text = re.sub(r"KOTAWeather", "KOTA Weather", text)
        text = re.sub(r"LuchaUnderground", "Lucha Underground", text)
        text = re.sub(r"KOIN6News", "KOIN 6 News", text)
        text = re.sub(r"LiveOnK2", "Live On K2", text)
        text = re.sub(r"9NewsGoldCoast", "9 News Gold Coast", text)
        text = re.sub(r"nikeplus", "nike plus", text)
        text = re.sub(r"david_cameron", "David Cameron", text)
        text = re.sub(r"peterjukes", "Peter Jukes", text)
        text = re.sub(r"JamesMelville", "James Melville", text)
        text = re.sub(r"megynkelly", "Megyn Kelly", text)
        text = re.sub(r"cnewslive", "C News Live", text)
        text = re.sub(r"JamaicaObserver", "Jamaica Observer", text)
        text = re.sub(r"textLikeItsSeptember11th2001", "text like it is september 11th 2001", text)
        text = re.sub(r"cbplawyers", "cbp lawyers", text)
        text = re.sub(r"fewmoretexts", "few more texts", text)
        text = re.sub(r"BlackLivesMatter", "Black Lives Matter", text)
        text = re.sub(r"cjoyner", "Chris Joyner", text)
        text = re.sub(r"ENGvAUS", "England vs Australia", text)
        text = re.sub(r"ScottWalker", "Scott Walker", text)
        text = re.sub(r"MikeParrActor", "Michael Parr", text)
        text = re.sub(r"4PlayThursdays", "Foreplay Thursdays", text)
        text = re.sub(r"TGF2015", "Tontitown Grape Festival", text)
        text = re.sub(r"realmandyrain", "Mandy Rain", text)
        text = re.sub(r"GraysonDolan", "Grayson Dolan", text)
        text = re.sub(r"ApolloBrown", "Apollo Brown", text)
        text = re.sub(r"saddlebrooke", "Saddlebrooke", text)
        text = re.sub(r"TontitownGrape", "Tontitown Grape", text)
        text = re.sub(r"AbbsWinston", "Abbs Winston", text)
        text = re.sub(r"ShaunKing", "Shaun King", text)
        text = re.sub(r"MeekMill", "Meek Mill", text)
        text = re.sub(r"TornadoGiveaway", "Tornado Giveaway", text)
        text = re.sub(r"GRupdates", "GR updates", text)
        text = re.sub(r"SouthDowns", "South Downs", text)
        text = re.sub(r"braininjury", "brain injury", text)
        text = re.sub(r"auspol", "Australian politics", text)
        text = re.sub(r"PlannedParenthood", "Planned Parenthood", text)
        text = re.sub(r"calgaryweather", "Calgary Weather", text)
        text = re.sub(r"weallheartonedirection", "we all heart one direction", text)
        text = re.sub(r"edsheeran", "Ed Sheeran", text)
        text = re.sub(r"TrueHeroes", "True Heroes", text)
        text = re.sub(r"S3XLEAK", "sex leak", text)
        text = re.sub(r"ComplexMag", "Complex Magazine", text)
        text = re.sub(r"TheAdvocateMag", "The Advocate Magazine", text)
        text = re.sub(r"CityofCalgary", "City of Calgary", text)
        text = re.sub(r"EbolaOutbreak", "Ebola Outbreak", text)
        text = re.sub(r"SummerFate", "Summer Fate", text)
        text = re.sub(r"RAmag", "Royal Academy Magazine", text)
        text = re.sub(r"offers2go", "offers to go", text)
        text = re.sub(r"foodscare", "food scare", text)
        text = re.sub(r"MNPDNashville", "Metropolitan Nashville Police Department", text)
        text = re.sub(r"TfLBusAlerts", "TfL Bus Alerts", text)
        text = re.sub(r"GamerGate", "Gamer Gate", text)
        text = re.sub(r"IHHen", "Humanitarian Relief", text)
        text = re.sub(r"spinningbot", "spinning bot", text)
        text = re.sub(r"ModiMinistry", "Modi Ministry", text)
        text = re.sub(r"TAXIWAYS", "taxi ways", text)
        text = re.sub(r"Calum5SOS", "Calum Hood", text)
        text = re.sub(r"po_st", "po.st", text)
        text = re.sub(r"scoopit", "scoop.it", text)
        text = re.sub(r"UltimaLucha", "Ultima Lucha", text)
        text = re.sub(r"JonathanFerrell", "Jonathan Ferrell", text)
        text = re.sub(r"aria_ahrary", "Aria Ahrary", text)
        text = re.sub(r"rapidcity", "Rapid City", text)
        text = re.sub(r"OutBid", "outbid", text)
        text = re.sub(r"lavenderpoetrycafe", "lavender poetry cafe", text)
        text = re.sub(r"EudryLantiqua", "Eudry Lantiqua", text)
        text = re.sub(r"15PM", "15 PM", text)
        text = re.sub(r"OriginalFunko", "Funko", text)
        text = re.sub(r"rightwaystan", "Richard Tan", text)
        text = re.sub(r"CindyNoonan", "Cindy Noonan", text)
        text = re.sub(r"RT_America", "RT America", text)
        text = re.sub(r"narendramodi", "Narendra Modi", text)
        text = re.sub(r"BakeOffFriends", "Bake Off Friends", text)
        text = re.sub(r"TeamHendrick", "Hendrick Motorsports", text)
        text = re.sub(r"alexbelloli", "Alex Belloli", text)
        text = re.sub(r"itsjustinstuart", "Justin Stuart", text)
        text = re.sub(r"gunsense", "gun sense", text)
        text = re.sub(r"DebateQuestionsWeWantToHear", "debate questions we want to hear", text)
        text = re.sub(r"RoyalCarribean", "Royal Carribean", text)
        text = re.sub(r"samanthaturne19", "Samantha Turner", text)
        text = re.sub(r"JonVoyage", "Jon Stewart", text)
        text = re.sub(r"renew911health", "renew 911 health", text)
        text = re.sub(r"SuryaRay", "Surya Ray", text)
        text = re.sub(r"pattonoswalt", "Patton Oswalt", text)
        text = re.sub(r"minhazmerchant", "Minhaz Merchant", text)
        text = re.sub(r"TLVFaces", "Israel Diaspora Coalition", text)
        text = re.sub(r"pmarca", "Marc Andreessen", text)
        text = re.sub(r"pdx911", "Portland Police", text)
        text = re.sub(r"jamaicaplain", "Jamaica Plain", text)
        text = re.sub(r"Japton", "Arkansas", text)
        text = re.sub(r"RouteComplex", "Route Complex", text)
        text = re.sub(r"INSubcontinent", "Indian Subcontinent", text)
        text = re.sub(r"NJTurnpike", "New Jersey Turnpike", text)
        text = re.sub(r"Politifiact", "PolitiFact", text)
        text = re.sub(r"Hiroshima70", "Hiroshima", text)
        text = re.sub(r"GMMBC", "Greater Mt Moriah Baptist Church", text)
        text = re.sub(r"versethe", "verse the", text)
        text = re.sub(r"TubeStrike", "Tube Strike", text)
        text = re.sub(r"MissionHills", "Mission Hills", text)
        text = re.sub(r"ProtectDenaliWolves", "Protect Denali Wolves", text)
        text = re.sub(r"NANKANA", "Nankana", text)
        text = re.sub(r"SAHIB", "Sahib", text)
        text = re.sub(r"PAKPATTAN", "Pakpattan", text)
        text = re.sub(r"Newz_Sacramento", "News Sacramento", text)
        text = re.sub(r"gofundme", "go fund me", text)
        text = re.sub(r"pmharper", "Stephen Harper", text)
        text = re.sub(r"IvanBerroa", "Ivan Berroa", text)
        text = re.sub(r"LosDelSonido", "Los Del Sonido", text)
        text = re.sub(r"bancodeseries", "banco de series", text)
        text = re.sub(r"timkaine", "Tim Kaine", text)
        text = re.sub(r"IdentityTheft", "Identity Theft", text)
        text = re.sub(r"AllLivesMatter", "All Lives Matter", text)
        text = re.sub(r"mishacollins", "Misha Collins", text)
        text = re.sub(r"BillNeelyNBC", "Bill Neely", text)
        text = re.sub(r"BeClearOnCancer", "be clear on cancer", text)
        text = re.sub(r"Kowing", "Knowing", text)
        text = re.sub(r"ScreamQueens", "Scream Queens", text)
        text = re.sub(r"AskCharley", "Ask Charley", text)
        
        # Urls
        text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)
            
        # Words with punctuations and special characters
        punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
        for p in punctuations:
            text = text.replace(p, f' {p} ')
            
        # ... and ..
        text = text.replace('...', ' ... ')
        if '...' not in text:
            text = text.replace('..', ' ... ')      
            
        # Acronyms
        text = re.sub(r"MH370", "Malaysia Airlines Flight 370", text)
        text = re.sub(r"mÌ¼sica", "music", text)
        text = re.sub(r"okwx", "Oklahoma City Weather", text)
        text = re.sub(r"arwx", "Arkansas Weather", text)    
        text = re.sub(r"gawx", "Georgia Weather", text)  
        text = re.sub(r"scwx", "South Carolina Weather", text)  
        text = re.sub(r"cawx", "California Weather", text)
        text = re.sub(r"tnwx", "Tennessee Weather", text)
        text = re.sub(r"azwx", "Arizona Weather", text)  
        text = re.sub(r"alwx", "Alabama Weather", text)
        text = re.sub(r"wordpressdotcom", "wordpress", text)    
        text = re.sub(r"usNWSgov", "United States National Weather Service", text)
        text = re.sub(r"Suruc", "Sanliurfa", text)   
        
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text=" ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text=" ".join(text)
        
        return text

    def predict_hate_speech(self, text):
        cleaned_text = self.clean_text(text)
        seq = self.load_tokenizer.texts_to_sequences([cleaned_text])
        padded = sequence.pad_sequences(seq, maxlen=300)
        pred = self.load_model.predict(padded)

        if pred < 0.3:
            return 0
        else:
            return 1

    def update_comments(self, urls, limit):
        data = [] 

        if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
            # El archivo existe y no está vacío, cargar el DataFrame desde el archivo
            df = pd.read_csv(self.data_file, sep=';')
            existing_comments = set(df['comment'].values)
        else:
            # El archivo no existe o está vacío, crear un DataFrame vacío
            df = pd.DataFrame(columns=['comment', 'date', 'user', 'hate'])
            existing_comments = set()

        # Obtener los nuevos comentarios usando la función get_comments_in_live_videos
        videos_ids = self.extract_videos_ids(urls)
        new_comments_df = self.get_comments_in_live_videos(videos_ids, limit=limit)

        # Filtrar los comentarios que son realmente nuevos
        new_comments_df = new_comments_df[~new_comments_df['comment'].isin(existing_comments)]
        new_comments = new_comments_df[['comment', 'date', 'user']]

        # Clasificar los comentarios nuevos usando la función predict_hate_speech
        new_comments['hate'] = new_comments['comment'].apply(self.predict_hate_speech)

        # Contar el número de '1' y '0' en la columna 'hate'
        hate_counts = df['hate'].value_counts().to_dict()
        count_1 = hate_counts.get(1, 0)
        count_0 = hate_counts.get(0, 0)

        # Imprimir los comentarios nuevos en la consola con el formato deseado, incluyendo la clasificación 'hate'
        for _, row in new_comments.iterrows():
            comment = row['comment']
            date = row['date']
            user = row['user']
            hate = row['hate']
            no_hate = 0

            if (hate == 0):
                no_hate = 1
                count_1 += 1
            else:
                count_0 += 1


            data.append({
                "comment": comment,
                "date": date,
                "user": user,
                "hate": hate,
                "no_hate": no_hate,
                "number_hate": count_1,
                "number_noHate": count_0,
                "group": "hate",
                "count": count_1 + count_0
            })

        # Concatenar y eliminar duplicados
        df_new = pd.concat([df, new_comments])
        df_new = df_new.drop_duplicates()

        # Guardar los comentarios actualizados en el archivo CSV
        df_new.to_csv(self.data_file, index=False, sep=';')

        # Devolver la lista de datos y los conteos de '1' y '0' en la columna 'hate'
        return data
