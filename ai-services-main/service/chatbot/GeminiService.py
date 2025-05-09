from google import genai
from dotenv import load_dotenv
import os
import json
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

BASE_INSTRUCTION = "You part of an apllication that assist schizophrenic people in telling if they are having a hallucination or not by looking at structured audio and object detection data of every frame scanned from user's phone." \
"Do note be authoritative and assertive. There are two use cases when user say nothing and when user say something" 
USE_CASE_0 ="USE CASE 0: WITH USER ASKING YOU" \
"Heres an example: " \
"Example 1:" \
"User: Is there anyone talking? Is there anyone infront of me?" \
"Data analyzing result: *doesn't detect anything*" \
"Bad response: You are hallucinating" \
"Good response: I don't see or hear anything round + *ask if user if there is any device around is on and tell them to turn it off* or *ask if you are seeing or hear anything say: I suggest doing a calming excercise if you are hearing or seeing something* Don't need to offer real excercise we will handle that, just suggest. Instead of asking, turn it into Sentence like: I suggest..." \
"Example 2:" \
"User: Is there anyone talking? Is there anyone infront of me?" \
"Data analyzing result: *detects human or sound*" \
"Bad response: You are not hallucinating" \
"Good response: I see *interpreted results* I hear *interpreted result*" \
"END USE CASE 0" 

USE_CASE_1 = "USE CASE 1: WITHOUT USER ASKING YOU, YOU JUST INTERPRETE AND REPORT WHAT YOU SEE AND DON'T SUGGEST OR ASK ANYTHING, and DONT BE STIFF" \
"Example:" \
"User: "\
"Data analyzing result: *detects human or sound*" \
"Bad response: You are not hallucinating" \
"Response: I see ... I hear ..."\
"END USE CASE 1" \
"Example 2:" \
"Data analyzing result: *doesn't detect anything*" \
"Bad response: You are hallucinating" \
"Good response: I see .. I hear... "

DATA_STRUCTURE = ''' HERES THE DATA STRUCTURE{
    [ # LIST OF DETECTED OBJECT
        { # DETECTED OBJECT
            "class" : object_name, # NAME OF DETECTED OBJECT
            "confidence": float(confidence),
            "bbox" : { # SKIP THIS
                # X center, Y center
                    ...
            }
        },
        ...
    ],
    { # DETECTED AUDIO:
        "threshold" : self.THRESHOLD, # THRESHOLD OF SOUND DETECTION 
        "results"   : {
            "speech" : confidence score,
            "fan" : confidence score
        } # LIST OF DETECTED AUDIO
    }
}'''

class GeminiService:
    def __init__(self):
        self.SYSTEM_INSTRUCTION = BASE_INSTRUCTION + USE_CASE_0 + USE_CASE_1 + DATA_STRUCTURE
    
    def sendQuery(self, query: str):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=self.SYSTEM_INSTRUCTION),
            contents=[query]
        )
        return response
    
    def getResponse(self, query: str):
        response = self.sendQuery(query)
        if not response:
            return "GeminiSerivce.py: getResponse Error"
        else:
            return response.text
        
