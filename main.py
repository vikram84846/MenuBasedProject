import speech_recognition as sr
import os
import time 
import pywhatkit
import pyautogui
import numpy as np
import matplotlib.pyplot as plt
from twilio.rest import Client
from instabot import Bot
import cv2
from pynput.keyboard import Key, Controller
from geopy.geocoders import Nominatim
keyboard = Controller()
from PIL import Image, ImageDraw
from googlesearch import search
import boto3
import tkinter as tk
from cvzone.HandTrackingModule import HandDetector
import random
import time
import pyttsx3
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from cvzone.HandTrackingModule import HandDetector
import boto3
import pandas
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
import pyttsx3
import webbrowser
import time
import pyautogui
from langchain.document_loaders import TextLoader
import threading
import time

def ec2_finger():
    
    def genOS():
        ec2=boto3.resource('ec2')
        instances= ec2.create_instances(MinCount=1, MaxCount=1, InstanceType="t2.micro", ImageId="ami-0ded8326293d3201b", SecurityGroupIds=['sg-0c7043809b8957ebd'])
        return instances[0].id

    def delOS(id):
        ec2=boto3.resource('ec2')
        ec2.instances.filter(InstanceIds=[id]).terminate()

    detector = HandDetector(maxHands=1 , detectionCon=0.8 )
    allOS=[]
    cap = cv2.VideoCapture(0)

    while True:
        ret,  photo = cap.read()
        hand = detector.findHands(photo , draw=False)
        if hand:
            detectHand = hand[0]
            if detectHand:
                fingerup = detector.fingersUp(detectHand)
                if detectHand['type'] == 'Left':
                    for i in fingerup:
                        if i==1:
                            allOS.append(genOS())

                else:
                    for i in fingerup:
                        if i==1:
                            delOS(allOS.pop())

        cv2.imshow("my photo", photo)
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def linearReg():
    
    dataset = pandas.read_csv("csv file")
    model = LinearRegression()
    y = dataset['marks']
    x = dataset['hrs']
    X = x.values.reshape(-1,1)
    model.fit(X,y)
    print("model prediction : ")
    print(model.predict([[3]]))
    print("model Coefficient : ")
    print(model.coef_)

def assistant():
    

    def speak(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def recognize_speech():
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print(f"Error with the speech recognition service; {e}")

        return None

    def open_whatsapp():
        speak("Opening WhatsApp.")
        webbrowser.open("https://web.whatsapp.com")
        time.sleep(15)  # Wait for 15 seconds to give you time to scan the QR code
        speak("WhatsApp is now open. You can use it on your browser.")

    if __name__ == "__main__":
        while True:
            recognized_text = recognize_speech()
            if recognized_text:
                if "google" in recognized_text:
                    speak("Opening Google.")
                    webbrowser.open("https://www.google.com")
                elif "youtube" in recognized_text:
                    speak("Opening YouTube.")
                    webbrowser.open("https://www.youtube.com")
                    # Wait for a moment before searching for music
                    speak("What music would you like to listen to?")
                    time.sleep(3)  # Wait for 3 seconds to give you time to respond
                    music_name = recognize_speech()
                    if music_name:
                        url = f"https://www.youtube.com/results?search_query={music_name}"
                        webbrowser.open(url)
                        time.sleep(5)  # Wait for the search results page to load
                        # Click on the first video link
                        try:
                            pyautogui.click(x=800, y=380)  # Adjust the coordinates as per your screen resolution
                        except pyautogui.FailSafeException:
                            print("Failed to click the video link. Please click it manually.")

                elif "python" in recognized_text and "code" in recognized_text:
                    speak("Opening Chrome and searching vimal daga.")
                    webbrowser.open("https://www.google.com/search?q=vimal+daga")  # Changed the search query
                elif "vimal daga" in recognized_text:  # Added a new condition to directly search for "vimal daga"
                    speak("Searching vimal daga on Google.")
                    webbrowser.open("https://www.google.com/search?q=vimal+daga")
                elif "whatsapp" in recognized_text:
                    open_whatsapp()
                elif "exit" in recognized_text or "stop" in recognized_text:
                    speak("Goodbye!")
                    break


def rekognition():
    client = boto3.client('rekognition',region_name='ap-south-1')
    with open("img to be fed",'rb') as imgFile:
        imgData=imgFile.read()
    response=client.detect_labels(Image={'Bytes':imgData},MaxLabels=8)
    response    
    labels= response["Labels"]
    labels
    for label in labels:
            print(f"Label: {label['Name']}, Confidence: {label['Confidence']:.2f}%")

def document_loader():
    loader = TextLoader(file_path="path")
    loader.file_path
    document = loader.load()
    document
    from langchain.text_splitter import CharacterTextSplitter
    textChunk = CharacterTextSplitter(chunk_size=200)
    texts = textChunk.split_documents(document)
    len(texts)
    myopenkey  = "openai key"
    from langchain.embeddings import OpenAIEmbeddings
    myembedmodel = OpenAIEmbeddings(openai_api_key=myopenkey)
    from langchain.vectorstores import Pinecone
    import pinecone
    pinecone.init(
            api_key="pinecode key",
            environment="enviornment"
    )
    docsearch=Pinecone.from_documents(
                    documents = texts,
                    embedding = myembedmodel,
                    index_name = 'name of the index'        
    )
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    qa =  RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key= myopenkey),
            chain_type="stuff",
            retriever=docsearch.as_retriever()
    )
    myquery = " your querry"
    qa({"query": myquery}) 

def cartoon():

    def cartoonize_image(image, gray_mode=False):
        # Convert image to grayscale
        if gray_mode:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply median blur to reduce noise and smooth the image
        gray = cv2.medianBlur(gray, 5)

        # Detect edges in the image using adaptive thresholding
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

        # Create a color version of the image
        color = cv2.bilateralFilter(image, 9, 300, 300)

        # Combine the edges with the color image using a bitwise AND operation
        cartoon = cv2.bitwise_and(color, color, mask=edges)

        return cartoon

    def cartoonize_video():
        # Start video capture
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a more intuitive selfie view
            frame = cv2.flip(frame, 1)

            # Apply cartoonize effect to the frame
            cartoon_frame = cartoonize_image(frame)

            # Show the original and cartoonized frames side by side
            stacked_frames = np.hstack((frame, cartoon_frame))
            cv2.imshow("Cartoonizer", stacked_frames)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        cartoonize_video()
 

def simaltanous():
    def function1():
        while True:
            print("aaaaaa")
            time.sleep(1)
    def function2():
        while True:
            print("bbbbbb")
            time.sleep(1)
    thread1 = threading.Thread(target=function1)
    thread1.start()
    thread2= threading.Thread( target=function2)
    thread2.start()

def coffeeMaker():
    model = HandDetector()
    import random as rdm
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')  # getting details of current voice
    engine.setProperty('voice',voices[1].id)
    cap = cv2.VideoCapture(0)
    music_list = ['ishare tere', 'jab koi baat', 'dil meri na sune', 'calm down', 'dheere dheere se meri jingdi',
                              'kya baat hai', 'illegal weapon 2.0']

    pyttsx3.speak("Good Evening ")

    while True:
        try:
            status , photo = cap.read()
            cv2.imshow("hi", photo)
            if cv2.waitKey(100) == 13:
                break
            #test

            flag = 0
            #test
            hand = model.findHands(photo , draw=False )
            if hand:

                lmlist = hand[0]
                fingeruplist = model.fingersUp(lmlist)
                print(fingeruplist)
                if fingeruplist is not None:
                    if (flag ==0):
                        pyttsx3.speak("Welcome to the Coffee shop, Tell me quantity you want ? ")
                        flag=flag+1
                    time.sleep(2)
                if fingeruplist == [0 ,1 , 0, 0 , 0] : 
                    pyttsx3.speak("Your single Coffer is preparing, Please wait for 10 min ")
                    pyttsx3.speak("Let's have some music")
                    #play music
                    music_selected = rdm.choice(music_list)
                    pywhatkit.playonyt(music_selected)                    
                    time.sleep(1)
                elif fingeruplist == [ 0 , 1 , 1 , 0 ,0 ]:
                    pyttsx3.speak("we are preparing two coffee for you please wait, Have a great Day ")
                    pyttsx3.speak("if you want to listen some music show me ok sign")
                    #play music    
                    music_selected = rdm.choice(music_list)
                    pywhatkit.playonyt(music_selected)
                    time.sleep(1)
                elif fingeruplist == [0,1,1,1,0]:
                    pyttsx3.speak("we are preparing three coffee for you please wait, Have a great Day ")

                else:
                    pass
            fingeruplist=None

        except:
            print("exception")
            pass


    cv2.destroyAllWindows()

    cap.release()

from tkinter import *

def videoDownload():
    from pyyoutube import Api
    from pytube import YouTube
    from threading import Thread
    from tkinter import messagebox


    def get_list_videos():
        global playlist_item_by_id
        # Clear ListBox
        list_box.delete(0, 'end')

        # Create API Object
        api = Api(api_key='appi key')

        if "youtube" in playlistId.get():
            playlist_id = playlistId.get()[len(
                "https://www.youtube.com/playlist?list="):]
        else:
            playlist_id = playlistId.get()

        # Get list of video links
        playlist_item_by_id = api.get_playlist_items(
            playlist_id=playlist_id, count=None, return_json=True)

        # Iterate through all video links and insert into listbox
        for index, videoid in enumerate(playlist_item_by_id['items']):
            list_box.insert(
                END, f" {str(index+1)}. {videoid['contentDetails']['videoId']}")

        download_start.config(state=NORMAL)


    def threading():
        # Call download_videos function
        t1 = Thread(target=download_videos)
        t1.start()


    def download_videos():
        download_start.config(state="disabled")
        get_videos.config(state="disabled")

        # Iterate through all selected videos
        for i in list_box.curselection():
            videoid = playlist_item_by_id['items'][i]['contentDetails']['videoId']

            link = f"https://www.youtube.com/watch?v={videoid}"

            yt_obj = YouTube(link)

            filters = yt_obj.streams.filter(progressive=True, file_extension='mp4')

            # download the highest quality video
            filters.get_highest_resolution().download()

        messagebox.showinfo("Success", "Video Successfully downloaded")
        download_start.config(state="normal")
        get_videos.config(state="normal")


    # Create Object
    root = Tk()
    # Set geometry
    root.geometry('400x400')

    # Add Label
    Label(root, text="Youtube Playlist Downloader",
        font="italic 15 bold").pack(pady=10)
    Label(root, text="Enter Playlist URL:-", font="italic 10").pack()

    # Add Entry box
    playlistId = Entry(root, width=60)
    playlistId.pack(pady=5)

    # Add Button
    get_videos = Button(root, text="Get Videos", command=get_list_videos)
    get_videos.pack(pady=10)

    # Add Scrollbar
    scrollbar = Scrollbar(root)
    scrollbar.pack(side=RIGHT, fill=BOTH)
    list_box = Listbox(root, selectmode="multiple")
    list_box.pack(expand=YES, fill="both")
    list_box.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=list_box.yview)

    download_start = Button(root, text="Download Start",
                            command=threading, state=DISABLED)
    download_start.pack(pady=10)

    # Execute Tkinter
    root.mainloop()

def pomodoro():
    import tkinter as tk
    import time

    # Create the main application window
    root = tk.Tk()
    root.title("Pomodoro Timer")
    root.geometry("300x200")
    root.configure(bg="#f0f0f0")

    # Initialize pomodoro_active flag
    pomodoro_active = False

    # Define Pomodoro functions
    def start_pomodoro():
        work_time = 25 * 60
        short_break_time = 5 * 60
        long_break_time = 15 * 60
        num_work_sessions = 4

        global pomodoro_active
        pomodoro_active = True

        while pomodoro_active and num_work_sessions > 0:
            countdown(work_time, "Work Time")
            if pomodoro_active:
                countdown(short_break_time, "Short Break Time")
                num_work_sessions -= 1

        if pomodoro_active:
            countdown(long_break_time, "Long Break Time")

        pomodoro_active = False
        timer_label.config(text="Pomodoro Stopped", fg="red")

    def stop_pomodoro():
        global pomodoro_active
        pomodoro_active = False
        timer_label.config(text="Pomodoro Stopped", fg="red")

    def countdown(seconds, session_type):
        global pomodoro_active
        while seconds and pomodoro_active:
            mins, secs = divmod(seconds, 60)
            timer_label.config(text=f"{session_type}\n{mins:02d}:{secs:02d}", fg="black")
            root.update()
            time.sleep(1)
            seconds -= 1
        if pomodoro_active:
            timer_label.config(text="Session Complete!", fg="green")
            root.update()
            time.sleep(2)
            timer_label.config(text="")
            root.update()

    # Create and position the buttons
    button_pomodoro = tk.Button(root, text="Start Pomodoro", command=start_pomodoro, padx=10, pady=5, bg="#ff9800", fg="white")
    button_pomodoro.pack(pady=20)

    button_stop_pomodoro = tk.Button(root, text="Stop Pomodoro", command=stop_pomodoro, padx=10, pady=5, bg="#e91e63", fg="white")
    button_stop_pomodoro.pack(pady=10)

    timer_label = tk.Label(root, text="", font=("Helvetica", 20), bg="#f0f0f0")
    timer_label.pack()

    # Start the main event loop
    root.mainloop()

def lanchain_dalle():
    inp = input("Enter your prompt : ")
    key= "key"
    myllm = OpenAI(
    model = 'text-davinci-003',
    temperature=1,
    openai_api_key=key
    )
    os.environ['OPENAI_API_KEY'] = key
    tools = load_tools(['dalle-image-generator'])
    agent = initialize_agent(tools, llm =myllm, agent="zero-shot-react-description", verbose=True)
    output = agent.run(inp)

def rock_paper_sci():
    cap = cv2.VideoCapture(0)  
    detector = HandDetector()
    sp = pyttsx3.init()

    gestures = ["rock", "paper", "scissors"]

    def detect_user_gesture(img):
        img1 = detector.findHands(img, draw = False)

        if img1:
            lmlist = img1[0]
            handphoto = detector.fingersUp(lmlist)
            return img, handphoto
        else:
            return None, None

    def get_computer_gesture():
        return random.choice(gestures)


    def get_winner(user_gesture, computer_gesture):

        if user_gesture == computer_gesture:
            sp.say("its a tie")
            sp.runAndWait()
            return "It's a tie!"

        elif (user_gesture == "rock" and computer_gesture == "scissors") or \
             (user_gesture == "paper" and computer_gesture == "rock") or \
             (user_gesture == "scissors" and computer_gesture == "paper"):
            sp.say("You Win")
            sp.runAndWait()

            return "You win!"

        else:
            sp.say("Computer Wins")
            sp.runAndWait()

            return "Computer wins!"

    def findGestures(lm, img):
        if lm == [0,0,0,0,0]:
            sp.say("Rock")
            sp.runAndWait()
            return "rock"
        elif lm == [1,1,1,1,1]:
            sp.say("Paper")
            sp.runAndWait()
            return "paper"
        elif lm == [0, 1, 1, 0, 0]:
            sp.say("scissor")
            sp.runAndWait()
            return 'scissors'
        else:
            return None

    while True:
        status , img = cap.read()

        cv2.imshow("myphoto",img)
        user_img, user_lmList = detect_user_gesture(img)

        computer_gesture = get_computer_gesture()

        user_gesture = findGestures(user_lmList, img)
        if  user_gesture!=None:
            print(user_gesture, computer_gesture.capitalize())
            winner = get_winner(user_gesture, computer_gesture)
            print(user_img, winner)
            time.sleep(2)


        if cv2.waitKey(10) == 13:
            break
    cv2.destroyAllWindows()


    cap.release()

def open_software(software_name):
    software_path = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
        "chrome":"chrome.exe",
        "command prompt":"cmd.exe",
        "explorer":"explorer.exe",
        "vlc":"vlc.exe",
         "taskmgr":"taskmgr",
        # Add more software names and paths here
    }

    if software_name in software_path:
        try:
            os.startfile(software_path[software_name])
        except Exception as e:
            status_label.config(text=f"Error: {e}")
    else:
        status_label.config(text="Software not found.")
    pass

def whatsapp():
    from pynput.keyboard import Key, Controller
    keyboard = Controller()
    try:
        pywhatkit.sendwhatmsg_instantly(
            phone_no="enter phone no.", 
            message="Hello from sachin",
            tab_close=True
        )
        time.sleep(20)
        pyautogui.click()
        time.sleep(5)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        print("Message sent!")
    except Exception as e:
        print(str(e))
        
        
def message():

        client = Client("id", "key")
        client.messages.create(to="", 
                               from_="", 
                               body="Hello Linux World!")


def click_photo():

   cap=cv2.VideoCapture(0)
   cap
   status ,photo =cap.read()
   cv2.imwrite("pic.jpg",photo)
   cv2.imshow("My photo",photo)
   cv2.waitKey(5000)
   cv2.destroyAllWindows()
   cap.release()
    

def crop_pic():
   cap=cv2.VideoCapture(0)
   cap
   status ,photo =cap.read()
   cv2.imwrite("pic.jpg",photo)
   cv2.imshow("My photo",photo[200:540,200:430])
   cv2.waitKey(5000)
   cv2.destroyAllWindows()
   cap.release()
    
def face_swap():    
    pic1=cv2.imread("kohli.jpeg.jpg")
    pic3=cv2.imread("babar.png")
    pic3[100:280,220:370]=pic1[120:300,200:350]
    cv2.imshow("my photo",pic3)
    cv2.waitKey()
    cv2.destroyAllWindows()    
    
def capture_video():
    cap=cv2.VideoCapture(0)
    while True:
        status ,photo=cap.read()
        cv2.imshow("My photo",photo)
        if cv2.waitKey(5)==13:
            break
    cv2.destroyAllWindows()

def capture_crop_video():
    cap=cv2.VideoCapture(0)
    while True:
        status ,photo=cap.read()
        photo[0:200,0:200]=photo[200:400,200:400]
        cv2.imshow("My photo",photo)
        if cv2.waitKey(5)==13:
            break
    cv2.destroyAllWindows()

    
def image_100_100():
    # Create a blank canvas for the image
    width = 400
    height = 300
    channels = 3
    image = np.zeros((height, width, channels), dtype=np.uint8)

    #Background (lemon)
    image[:300, 0:400, 0] = 255
    image[:300, 0:400, 1] = 255
    image[:300, 0:400, 2] = 102

    # Table (Brown)
    #1st leg

    image[200:300, 50:100 , 0] = 55
    image[200:300, 50:100, 1] = 0
    image[200:300, 50:100, 2] = 9

    #2nd leg

    image[200:300, 300:350 , 0] = 55
    image[200:300, 300:350, 1] = 0
    image[200:300,300:350, 2] = 9

    # Surface
    image[175:200,25:375, 0] = 55
    image[175:200,25:375, 1] = 0
    image[175:200,25:375, 2] = 9


    # TV (Black)
    # base (black)

    image[160:175,160:240, 0] = 0
    image[160:175,160:240, 1] = 0
    image[160:175,160:240, 2] = 0

    # Screen back
    image[50:160, 100:300,0] = 0
    image[50:160, 100:300,1] = 0
    image[50:160, 100:300,2] = 0

    # Screen(view) (sky blue)

    image[60:150, 110:290, 0] = 108
    image[60:150, 110:290, 1] = 255
    image[60:150, 110:290, 2] = 255
    # Display the image
    plt.imshow(image)
    plt.axis('on')
    plt.show()



    
def get_coordinates():
    location_name = input("enter the city name:")
    geolocator = Nominatim(user_agent="location_finder")
    location = geolocator.geocode(location_name)
    if location is None:
        print(f"Coordinates not found for '{location_name}'.")
        return None
    else:
        latitude = location.latitude
        longitude = location.longitude
        print(f"Coordinates for '{location_name}': Latitude = {latitude}, Longitude = {longitude}.")
        return latitude, longitude

    # Replace 'New York City' with your desired location.
    location_name = input("enter the city name:")
    

def top_10_google_searches():

    query = input("Enter what you want to search: ")
    result = int(input("How many results you want: "))

    for i in search(query, num=result, stop=result, pause=2):
        print(i)
        
def instabot():
    # Set your Instagram username and password
    username = "your user name"
    password = "your passwd"

    # Create an instance of the Instabot class
    bot = Bot()

    # Log in to Instagram
    bot.login(username=username, password=password, use_cookie=False, ask_for_code=True)

    # Open the image and resize it to a square (1:1) aspect ratio
    image_path = "img" # Use forward slashes in the path
    image = Image.open(image_path)
    width, height = image.size
    min_dimension = min(width, height)
    resized_image = image.crop((0, 0, min_dimension, min_dimension))

    # Save the resized image to a temporary file
    temp_image_path = "temp.jpg"
    resized_image.save(temp_image_path)

    # Upload the resized image with a caption
    caption = "failure is the best teacher if you fail in right direction then you will acheive success!"
    bot.upload_photo(temp_image_path, caption=caption)

    # Logout from your account
    bot.logout()

        
def launch_instance():
    launch = boto3.client('ec2',region_name='ap-south-1')
    launch.run_instances(
        ImageId='ami-0da59f1af71ea4ad2',
        InstanceType='t2.micro',
        MaxCount=1,
        MinCount=1
        )
    describe_instance = boto3.client('ec2')
    describe_instance.describe_instances()
    
def create_bucket():
    bucket = boto3.client('s3',region_name='ap-south-1')
    bucket.create_bucket(
    Bucket='it should be unique worldwide',
    ACL='private',
    CreateBucketConfiguration={
          'LocationConstraint': 'ap-south-1'}
    )
    
    
def use_sns_service():
    sns = boto3.client('sns',region_name='ap-south-1')
    sns.publish(
    Message='Dont take it serious.',
    Subject='this is automatd sns service.',
    TopicArn='arn:aws:sns:ap-south-1:299592517672:python_menu'
    )
    print("email sent")
    

def clear_status():
    status_label.config(text="cleared")

def create_button(parent, label, command):
    button = tk.Button(parent,font=("Arial",10,"bold"), text=label,width=20,height=2, command=command)
    return button


root = tk.Tk()
root.title("Main Window")
root.geometry("1200x900")
root.configure(bg="Black")
software_entry = tk.Entry(root,width=64)
software_entry.pack(pady=20)


buttons_frame = tk.Frame(root, bg="orange")
buttons_frame.pack(padx=20, pady=20, fill="both", expand=True)


button_notepad = create_button(buttons_frame, "Video Downloader",videoDownload)
button_calculator = create_button(buttons_frame, "EC2 WITH HANDS", ec2_finger)
button_paint = create_button(buttons_frame, "Voice Assistant", assistant)
button_chrome = create_button(buttons_frame, "Two Functions", simaltanous)
button_face_swap = create_button(buttons_frame, "FACE SWAP", face_swap)
button_explorer = create_button(buttons_frame, "Coffee Maker", coffeeMaker)
button_vlc = create_button(buttons_frame, "Rekognition", rekognition)
button_instabot = create_button(buttons_frame, "INSTABOT",instabot)
button_whatsapp = create_button(buttons_frame, "SEND WHATSAPP", whatsapp)
button_message = create_button(buttons_frame, "SEND MESSAGE", message)
button_photo = create_button(buttons_frame, "CLICK PHOTO",click_photo)
button_croppic = create_button(buttons_frame, "CROP PHOTO",crop_pic)
button_video = create_button(buttons_frame, "CAPTURE VIDEO",capture_video)
button_cropvideo = create_button(buttons_frame,"CROP VIDEO",capture_crop_video)
button_image= create_button(buttons_frame,"IMAGE_CREATION",image_100_100)
button_coordinates = create_button(buttons_frame,"GEO COORDINATES" ,lambda:get_coordinates())
button_searchresults = create_button(buttons_frame,"TOP10GOOGLESEARCHES",lambda:top_10_google_searches())
button_launchinstance = create_button(buttons_frame,"LAUNCH INSTANCE",launch_instance)
button_createbucket = create_button(buttons_frame,"CREATE BUCKET",create_bucket)
button_usesnsservice = create_button(buttons_frame,"USE SNS SERVICE",use_sns_service)
button_rock_paper = create_button(buttons_frame, "Play Rock Paper Scissor", rock_paper_sci)
button_dalle = create_button(buttons_frame, "Dalle e 2", lanchain_dalle)
button_linear = create_button(buttons_frame, "Linear Regression", linearReg)
button_load = create_button(buttons_frame, "Document Loader", document_loader)
button_cartoon = create_button(buttons_frame, "Pomodoro", pomodoro)




button_notepad.grid(row=0, column=0, padx=20, pady=40)
button_calculator.grid(row=0, column=1, padx=20, pady=20)
button_paint.grid(row=0, column=2, padx=30, pady=20)
button_chrome.grid(row=0, column=3, padx=10, pady=20)
button_face_swap.grid(row=1, column=0, padx=20, pady=20)
button_explorer.grid(row=1, column=1, padx=30, pady=20)
button_vlc.grid(row=1, column=2, padx=10, pady=20)
button_instabot.grid(row=1, column=3, padx=10, pady=20)
button_whatsapp.grid(row=2, column=0, padx=20, pady=20)
button_message.grid(row=2, column=1, padx=30, pady=20)
button_photo.grid(row=2, column=2, padx=20, pady=20)
button_croppic.grid(row=2, column=3, padx=30, pady=20)
button_video.grid(row=3, column=0, padx=40, pady=20)
button_cropvideo.grid(row=3, column=1, padx=50, pady=20)
button_image.grid(row=3, column=2,padx=40, pady=20)
button_coordinates.grid(row=3, column=3, padx=50, pady=20)
button_searchresults.grid(row=4, column=0, padx=40, pady=20) 
button_launchinstance.grid(row=4, column=1, padx=40, pady=20)
button_createbucket.grid(row=4, column=2, padx=40, pady=20)
button_usesnsservice.grid(row=4, column=3, padx=40, pady=20)
button_rock_paper.grid(row=5, column=0, padx=40, pady=20)
button_dalle.grid(row=5, column=1, padx=40, pady=20)
button_linear.grid(row=5, column=2, padx=40, pady=20)
button_load.grid(row=5, column=3, padx=40, pady=20)
button_cartoon.grid(row=6, column=0, padx=40, pady=20)

status_label = tk.Label(root, text="", fg="red")
status_label.pack(pady=10)

clear_button = tk.Button(root, text="Clear Status",fg="Red",font=("Arial", 20, "bold"),width=25 ,command=clear_status)
clear_button.pack()

root.mainloop()
