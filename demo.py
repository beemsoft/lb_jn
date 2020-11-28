import os
import time

import cv2
import face_recognition
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from pylgbst import *
from pylgbst.hub import MoveHub
from pylgbst.peripherals import EncodedMotor

speak = False
name = ''
nameOld = ''
video_capture = None
movehub = None
known_face_encodings = []
known_face_names = []
unknown_face_limit = 6
unknown_face_count = 0
wave_limit = 5
wave_count = 0
recognizer = sr.Recognizer()
log = logging.getLogger("demo")


def say_text(sentence):
    global speak
    speak = True
    output = gTTS(text=sentence, lang='en', slow=False)
    output.save('output.mp3')
    os.system('mpg123 output.mp3')
    speak = False


def wave_callback(color, distance=None):
    global wave_count
    global nameOld

    wave_count += 1
    log.info("Wave detected at distance: " + str(distance))

    if wave_count > wave_limit:
        nameOld = ""
        wave_count = 0


def shake_head():
    motor = None
    if isinstance(movehub.port_D, EncodedMotor):
        motor = movehub.port_D
    elif isinstance(movehub.port_C, EncodedMotor):
        motor = movehub.port_C
    else:
        log.warning("Motor not found on ports C or D")

    if motor:
        motor.angled(20, 0.4)
        motor.angled(40, -0.4)
        motor.angled(20, 0.4)


def dramatic_turn():
    movehub.motor_AB.timed(0.60, 0.2, -0.2)
    time.sleep(1)
    movehub.motor_AB.timed(0.60, 0.2, -0.2)
    time.sleep(1)
    movehub.motor_AB.timed(0.60, 0.2, -0.2)
    time.sleep(2)
    movehub.motor_AB.timed(0.9, -0.4, 0.4)


def ask_name():
    global name
    say_text('I see a new face! What\'s your name?')
    try:
        with sr.Microphone() as source:
            print("Say something!")
            audio = recognizer.listen(source, timeout=5)
            name = recognizer.recognize_google(audio, None, "en")
            print("Google Speech Recognition thinks you said " + name)
            return name
    except sr.WaitTimeoutError:
        print("Google Speech Recognition timeout")
        say_text('Please repeat that in the mic')
        return ask_name()
    except sr.UnknownValueError:
        print("Google Speech Recognition unknown value")
        say_text('Please repeat that in the mic')
        return ask_name()
    except sr.RequestError as e:
        say_text('I could not recognize that')
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720,
                                framerate=60, flip_method=0):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the RPI camera on a Jetson Nano
    """
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip_method} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    )


def train_faces():
    global known_face_encodings
    global known_face_names
    print("Training for face recognition")
    harry_image = face_recognition.load_image_file("harry.jpg")
    harry_face_encoding = face_recognition.face_encodings(harry_image)[0]
    legolas_image = face_recognition.load_image_file("legolas.jpeg")
    legolas_face_encoding = face_recognition.face_encodings(legolas_image)[0]
    known_face_encodings = [harry_face_encoding, legolas_face_encoding]
    known_face_names = ["Harry Potter", "Legolas"]


def add_face(face_encoding):
    new_name = ask_name()
    if new_name != "stop":
        known_face_encodings.append(face_encoding)
        known_face_names.append(new_name)
        say_text('Nice to meet you, ' + new_name)


def do_image_recognition():
    global speak
    global nameOld
    global video_capture
    global unknown_face_count
    process_this_frame = True

    # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
    video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        # Also, don't process during speaking
        if process_this_frame and speak is False:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    known_face_name = known_face_names[best_match_index]
                    if known_face_name != nameOld:
                        shake_head()
                        say_text('Hello, ' + known_face_name)
                    nameOld = known_face_name
                else:
                    # If an unknown face has been detected for a few times, add it to the list
                    unknown_face_count += 1
                    if unknown_face_count > unknown_face_limit:
                        unknown_face_count = 0
                        add_face(face_encoding)

        process_this_frame = not process_this_frame


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(relativeCreated)d\t%(levelname)s\t%(name)s\t%(message)s')
    try:
        train_faces()
        say_text('Hi, I\'m Verny! Please push on the green button')
        movehub = MoveHub()
        movehub.vision_sensor.subscribe(wave_callback)
        dramatic_turn()
        do_image_recognition()
    finally:
        # Release handle to the webcam
        video_capture.release()
        say_text('Okay, goodbye!')
        movehub.disconnect()
