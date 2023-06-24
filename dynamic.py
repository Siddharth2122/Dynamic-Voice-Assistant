import datetime
import os
import webbrowser
import urllib.request,bs4 as bs,sys,threading
from annex import Annex
from tkinter import *
from ttkthemes import themed_tk
from PIL import ImageTk,Image
import tkinter as tk
from tkinter import scrolledtext
import pyttsx3 as ptx
import speech_recognition as sr
import wikipedia
from functools import partial
import wolframalpha
import chatbot

try:
    app = wolframalpha.Client("QWPT4R-L3VP42G72V")
except Exception:
    print("Some technical error")


engine = ptx.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice",voices)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def CommandsList():
    '''show the command to which voice assistant is registered with'''
    os.startfile('Commands List.txt')


def wish():
    hour = int(datetime.datetime.now().hour)
    if hour >=0 and hour<12:
        speak("good morning sir i am your virtual assistance dynamic....how may i help you")
    elif hour>=12 and hour<18:
        speak("good afternoon sir i am your virtual assistance dynamic....how may i help you")
    else:
        speak("good evening sir i am your virtual assistance dynamic....how may i help you")


def takeCom():
    r= sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.......")
        r.adjust_for_ambient_noise(source)
        audio=r.listen(source)
    try:
        print("Recognising.......")
        text=r.recognize_google(audio)
        print(text)
    except Exception:
        #speak("error occured")
        print("SOMETHING ERROR")
        return "none"
    return text


def takeComm():
    r= sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.......")
        r.adjust_for_ambient_noise(source)
        audio=r.listen(source)
    try:
        print("Recognising.......")
        text=r.recognize_google(audio)
        print(text)
    except Exception:
        print("SOMETHING ERROR")
        return "none"
    return text


if __name__=="__main__":
    wish()
    while True:
        query = takeComm().lower()
        if(("dynamic" or "hello" or "hi" or "hello dynamic") in query):
            query = takeCom().lower()
            speak("yes sir")
            if "wikipedia" in query:
                speak("searching details....wait")
                query.replace("wikipedia","")
                results = wikipedia.summary(query,sentences=2)
                print(results)
                speak(results)
                continue

            
            elif ("google" or " open google" ) in query:
                webbrowser.open("www.google.co.in")
                speak("opening google")
                continue


            elif ("you tube" or "youtube" or "open youtube") in query:
                webbrowser.open("www.youtube.com")
                speak("opening youtube")
                continue


            elif ("open facebook" or "facebook" or "face book") in query:
                webbrowser.open("www.facebook.com")
                speak("opening facebook")
                continue


            elif ("open instagram" or "instagram" ) in query:
                webbrowser.open("www.instagram.com")
                speak("opening instagram")
                continue


            elif ("open yahoo" or "yahoo") in query:
                webbrowser.open("www.yahoo.com")
                speak("opening yahoo")
                continue


            elif ("open gmail" or "gmail" or "g mail") in query:
                webbrowser.open("www.gmail.com")
                speak("opening gmail")
                continue


            elif ("open amazon" or "amazon") in query:
                webbrowser.open("www.amazon.com")
                speak("opening amazon")
                continue


            elif ("open flipkart" or "flipkart" or "flip kart") in query:
                webbrowser.open("www.flipkart.com")
                speak("opening flipkart")
                continue


            elif ("spotify" or "open spotify")  in query:
                speak("opening spotify")
                os.startfile("C:\\Users\\ASUS\\AppData\\Roaming\\Spotify\\Spotify.exe")
                continue


            elif ("whatsapp" or "open whatsapp" or "whats app") in query:
                speak("opening whatsapp")
                os.startfile("C:\\Users\\ASUS\\AppData\\Local\\WhatsApp\\WhatsApp.exe")
                continue


            elif ("draw something" or "draw" or "show me some 3d pictures" or "3d pictures") in query:
                speak("i have two pictures right now")
                speak("one is star pattern and second is triangle pattern")
                speak("choose one to draw")
                while (True):
                    query = takeComm().lower()
                    if ("star") in query:
                        import star
                        star.main()
                    elif ("triangle") in query:
                        import triangle
                        triangle.main()


            elif ("draw indian flag" or "indian flag" or "national flag" or "flag" or "draw national flag") in query:
                speak("here is our national flag")
                import flag
                flag.main()
                continue


            elif ("play music" or "open music") in query:
                speak("ok i am playing music please wait")
                music_dic = "D:\music"
                musics = os.listdir(music_dic)
                os.startfile(os.path.join(music_dic,musics[0]))
                continue


            elif ("sketch me" or "potrait background" or "blur" or "background") in query:
                speak("ok let's try something new and enjoy the background")
                import sketch
                sketch.main()
                continue


            elif ("i want to play game" or "play game" or "game" or "lets play a game") in query:
                speak("ok let's play")
                import snake
                snake.main()
                continue


            elif ("let's talk" or "can we talk" or "talk to me" or "talk") in query:
                speak("why not i am eager to talk to you")
                chatbot.main()
                continue


            elif ("temperature" or "what is the current temperature" ) in query:
                res = app.query(query)
                print(next(res.results).text)
                speak(next(res.results).text)
                continue
               

            elif ("time" or "what is the current time") in query:
                res = app.query(query)
                print(next(res.results).text)
                speak(next(res.results).text)
                continue


            elif ("date") in query:
                res = app.query(query)
                print(next(res.results).text)
                speak(next(res.results).text)
                continue
               

            elif ("goodbye" or "good bye" or "bye") in query:
                speak("good bye have a great day")
                exit()


            elif ("shut down" or "shutdown" or "shut down my pc") in query:
                speak("shutting down your pc")
                os.system("shutdown /s /t 1")
                continue


            else:
                try:
                    res = app.query(query)
                    print(next(res.results).text)
                    speak(next(res.results).text)
                except:
                    speak("internet connection error")
                    print("internet connection error")



