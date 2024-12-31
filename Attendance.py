import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import openpyxl
from openpyxl import Workbook, load_workbook
import pandas as pd
import os
import subprocess
import psutil
import json

def logFaceDistance(name, distance):
    """
    Logs the face distance for a given name into a JSON file.
    """
    file_path = 'face_distances.json'

    # Load existing data or create an empty dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Append the new distance to the existing data for the person
    if name in data:
        data[name].append(distance)
    else:
        data[name] = [distance]

    # Save the updated data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def getAverageFaceDistance(name):
    """
    Retrieves the average face distance for a given name from the JSON file.
    """
    file_path = 'face_distances.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if name in data:
            return sum(data[name]) / len(data[name])  # Calculate average
    return None  # Return None if no data exists for the name

def calculateAccuracy(faceDistance):
    """
    Maps face distance to an accuracy percentage based on refined thresholds.
    """
    if faceDistance <= 0.30:
        return 100  # Perfect match
    elif 0.30 < faceDistance <= 0.40:
        # Linearly decrease from 100% to 90%
        return 100 - ((faceDistance - 0.30) / 0.10) * 10
    elif 0.40 < faceDistance <= 0.50:
        # Linearly decrease from 90% to 70%
        return 90 - ((faceDistance - 0.40) / 0.10) * 20
    elif 0.50 < faceDistance <= 0.60:
        # Linearly decrease from 70% to 50%
        return 70 - ((faceDistance - 0.50) / 0.10) * 20
    elif 0.60 < faceDistance <= 0.70:
        # Linearly decrease from 50% to 30%
        return 50 - ((faceDistance - 0.60) / 0.10) * 20
    else:
        # Below 30% for distances greater than 0.70
        return max(0, 30 - ((faceDistance - 0.70) / 0.10) * 30)

def writeToExcelRealTime(name, time):
    file_path = 'logbook.xlsx'  # Define the file path for the Excel file

    if not os.path.exists(file_path):  # Check if the Excel file exists
        # Create a new DataFrame and save it if the file doesn't exist
        df = pd.DataFrame(columns=['Name', 'Time'])  # Create empty DataFrame with columns
        df.to_excel(file_path, index=False)  # Write empty DataFrame to Excel file

    # Load the existing Excel file
    df = pd.read_excel(file_path)  # Read the Excel file into a DataFrame

    # Append the new entry
    new_entry = pd.DataFrame({'Name': [name], 'Time': [time]}) # Create a dictionary for the new entry
    df = pd.concat([df,new_entry], ignore_index=True)  # Append the new entry to the DataFrame

    # Save the updated DataFrame back to the Excel file
    df.to_excel(file_path, index=False)  # Write updated DataFrame to the Excel file

    # Close any previously open instance of the logbook
    pkill("UltimateFileViewer.exe")
    pkill("Excel.exe")

    # Open the logbook to show the updated data
    openLogbook(file_path)

def pkill (process_name):
    try:
        killed = os.system('TASKKILL /F /IM ' + process_name)
    except Exception as e:
        killed = 0
    return

def openLogbook(file_path):
    """
    Opens the logbook (Excel file) to display the updated data.
    """
    try:
        # Open the Excel file (works on Windows and macOS)
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['open', file_path])
        else:
            print("Unsupported operating system.")
    except Exception as e:
        print(f"Error opening Excel: {e}")

path = 'ImagesAttendance' #create path
images = [] #create list of images
classNames = [] #create list of class names
myList = os.listdir(path) #list of all elements in the directory ImagesAttendance
#print(myList) #prints the list of all elems in dir

for cls in myList: #iterate through list of elements in directory
    curImg = cv2.imread(f'{path}/{cls}') #each image is read by their full file name
    images.append(curImg) #each image is added to images
    classNames.append(os.path.splitext(cls)[0]) #each image is added without extension to classNames

#print(classNames) #classNames is printed to console

def findEncodings(images):
    encodeList = []
    for img in images: #iterates through each image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert color of image from BGR to RGB
        encode = face_recognition.face_encodings(img)[0]  # encode detected face
        encodeList.append(encode) #adds each encoded image to list
    return encodeList

def markedAttendance(name):
    with open('Attendance.csv','r+') as f: #opens attendance csv file, for reading and writing
        myDataList = f.readlines() #reads in current lines
        nameList = [] #creates empty list
        for line in myDataList: #iterate through lines in list
            entry = line.split(',') #each new entry is set to the name separated by time
            nameList.append(entry[0]) #entries are added to nameList
        if name not in nameList: #if the name is not in the list
            now = datetime.now() #sets variable now to the current time
            dateString = now.strftime('%H:%M:%S') #formats current time
            f.writelines(f'\n{name},{dateString}') #writes to file the name and time of person scanned
            writeToExcelRealTime(name, dateString)  # Writes to Excel in real time


encodeListKnown = findEncodings(images) #list of encodings set to variable encodeListKnown
#print(len(encodeListKnown)) #prints number of encoded images

cap = cv2.VideoCapture(0) #set variable to capture video from webcam

while True:
    success, img = cap.read() #while True, captures images from webcam

    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25) #resize image
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)  # convert color of image from BGR to RGB

    facesCurrent = face_recognition.face_locations(imgSmall)  # faces in current frame
    encodesCurrent = face_recognition.face_encodings(imgSmall, facesCurrent)  # encodes currently detected face

    for encodeFace,faceLocation in zip(encodesCurrent,facesCurrent): #grabs faces location and current frame
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) #compares faces to all faces known
        faceDistance = face_recognition.face_distance(encodeListKnown,encodeFace) #finds distance of faces
        print(faceDistance) #prints distance of each face in the list
        matchIndex = np.argmin(faceDistance) #matches the lowest element in list, lowest distance

        if matches[matchIndex]: #if the face is matched,
            name = classNames[matchIndex].upper() #sets name of identified face to uppercase,
            print(f"{name} - Face Distance: {faceDistance[matchIndex]}")  #debugging output

            #retrieve average distance and calculate accuracy
            avg_distance = getAverageFaceDistance(name)
            accuracy = calculateAccuracy(faceDistance[matchIndex])

            y1,x2,y2,x1 = faceLocation #sets coordinates to face location
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 #rescale scanned image
            cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,0),2) #shows green rectangle over scanning face
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,f"{name} ({accuracy:.2f}%)",(x1+6,y2-6),cv2.FONT_ITALIC,1,(0,0,255),2) #shows text of name of scanned person

            markedAttendance(name) #marks attendance of person scanned


    cv2.imshow('Webcam', img) #shows webcam
    cv2.waitKey(1)


#imgWill = face_recognition.load_image_file('ImagesBasic/willsmith.jpg') #setting file to picture of Will Smith
#imgWill = cv2.cvtColor(imgWill, cv2.COLOR_BGR2RGB) #convert color from BGR to RGB
#imgTest = face_recognition.load_image_file('ImagesBasic/willtest.jpg') #setting file to picture of Will Smith for testing
#imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB) #convert color from BGR to RGB