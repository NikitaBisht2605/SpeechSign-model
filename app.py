import streamlit as st
import queue
import av
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode
)
from typing import List, NamedTuple
from PIL import Image, ImageOps
import cv2
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from io import BytesIO
import streamlit.components.v1 as components
import pandas as pd
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List
import pydub
import mediapipe as mp
import tensorflow as tf
# np.set_printoptions(suppress=True)


global query_param


def firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            'static/speechsign-23477-8f5b84f0980a.json')
        app = firebase_admin.initialize_app(cred)
    else:
        app = firebase_admin.get_app()
    db = firestore.client()
    return app, db


@st.cache()
def cache_query_param():
    query_param = st.experimental_get_query_params()
    print(query_param)
    user_id = query_param['user'][0]
    return user_id, query_param


def extract_feature(image):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:
        while True:
            results = hands.process(
                cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            image_height, image_width, _ = image.shape
            # Print handedness (left v.s. right hand).
            # Caution : Uncomment these print command will resulting long log of mediapipe log
            #print(f'Handedness of {input_image}:')
            # print(results.multi_handedness)

            # Draw hand landmarks of each hand.
            # Caution : Uncomment these print command will resulting long log of mediapipe log
            #print(f'Hand landmarks of {input_image}:')
            if not results.multi_hand_landmarks:
                # Here we will set whole landmarks into zero as no handpose detected
                # in a picture wanted to extract.

                # Wrist Hand
                wristX = 0
                wristY = 0
                wristZ = 0

                # Thumb Finger
                thumb_CmcX = 0
                thumb_CmcY = 0
                thumb_CmcZ = 0

                thumb_McpX = 0
                thumb_McpY = 0
                thumb_McpZ = 0

                thumb_IpX = 0
                thumb_IpY = 0
                thumb_IpZ = 0

                thumb_TipX = 0
                thumb_TipY = 0
                thumb_TipZ = 0

                # Index Finger
                index_McpX = 0
                index_McpY = 0
                index_McpZ = 0

                index_PipX = 0
                index_PipY = 0
                index_PipZ = 0

                index_DipX = 0
                index_DipY = 0
                index_DipZ = 0

                index_TipX = 0
                index_TipY = 0
                index_TipZ = 0

                # Middle Finger
                middle_McpX = 0
                middle_McpY = 0
                middle_McpZ = 0

                middle_PipX = 0
                middle_PipY = 0
                middle_PipZ = 0

                middle_DipX = 0
                middle_DipY = 0
                middle_DipZ = 0

                middle_TipX = 0
                middle_TipY = 0
                middle_TipZ = 0

                # Ring Finger
                ring_McpX = 0
                ring_McpY = 0
                ring_McpZ = 0

                ring_PipX = 0
                ring_PipY = 0
                ring_PipZ = 0

                ring_DipX = 0
                ring_DipY = 0
                ring_DipZ = 0

                ring_TipX = 0
                ring_TipY = 0
                ring_TipZ = 0

                # Pinky Finger
                pinky_McpX = 0
                pinky_McpY = 0
                pinky_McpZ = 0

                pinky_PipX = 0
                pinky_PipY = 0
                pinky_PipZ = 0

                pinky_DipX = 0
                pinky_DipY = 0
                pinky_DipZ = 0

                pinky_TipX = 0
                pinky_TipY = 0
                pinky_TipZ = 0

                # Set image to Zero
                annotated_image = 0

                # Return Whole Landmark and Image
                return (wristX, wristY, wristZ,
                        thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                        thumb_McpX, thumb_McpY, thumb_McpZ,
                        thumb_IpX, thumb_IpY, thumb_IpZ,
                        thumb_TipX, thumb_TipY, thumb_TipZ,
                        index_McpX, index_McpY, index_McpZ,
                        index_PipX, index_PipY, index_PipZ,
                        index_DipX, index_DipY, index_DipZ,
                        index_TipX, index_TipY, index_TipZ,
                        middle_McpX, middle_McpY, middle_McpZ,
                        middle_PipX, middle_PipY, middle_PipZ,
                        middle_DipX, middle_DipY, middle_DipZ,
                        middle_TipX, middle_TipY, middle_TipZ,
                        ring_McpX, ring_McpY, ring_McpZ,
                        ring_PipX, ring_PipY, ring_PipZ,
                        ring_DipX, ring_DipY, ring_DipZ,
                        ring_TipX, ring_TipY, ring_TipZ,
                        pinky_McpX, pinky_McpY, pinky_McpZ,
                        pinky_PipX, pinky_PipY, pinky_PipZ,
                        pinky_DipX, pinky_DipY, pinky_DipZ,
                        pinky_TipX, pinky_TipY, pinky_TipZ,
                        image)

            annotated_image = cv2.flip(image.copy(), 1)
            for hand_landmarks in results.multi_hand_landmarks:
                # Wrist Hand /  Pergelangan Tangan
                wristX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
                wristY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                wristZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                # Thumb Finger / Ibu Jari
                thumb_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
                thumb_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
                thumb_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z

                thumb_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
                thumb_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
                thumb_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z

                thumb_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
                thumb_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
                thumb_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z

                thumb_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
                thumb_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
                thumb_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

                # Index Finger / Jari Telunjuk
                index_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
                index_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
                index_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

                index_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
                index_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
                index_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

                index_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
                index_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
                index_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z

                index_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                index_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                index_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                # Middle Finger / Jari Tengah
                middle_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
                middle_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
                middle_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

                middle_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
                middle_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
                middle_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

                middle_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
                middle_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
                middle_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z

                middle_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
                middle_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
                middle_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

                # Ring Finger / Jari Cincin
                ring_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
                ring_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
                ring_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

                ring_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
                ring_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
                ring_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

                ring_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
                ring_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
                ring_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z

                ring_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
                ring_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
                ring_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

                # Pinky Finger / Jari Kelingking
                pinky_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
                pinky_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
                pinky_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z

                pinky_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
                pinky_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
                pinky_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z

                pinky_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
                pinky_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
                pinky_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z

                pinky_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
                pinky_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
                pinky_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

                # Draw the Skeleton
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            return (wristX, wristY, wristZ,
                    thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                    thumb_McpX, thumb_McpY, thumb_McpZ,
                    thumb_IpX, thumb_IpY, thumb_IpZ,
                    thumb_TipX, thumb_TipY, thumb_TipZ,
                    index_McpX, index_McpY, index_McpZ,
                    index_PipX, index_PipY, index_PipZ,
                    index_DipX, index_DipY, index_DipZ,
                    index_TipX, index_TipY, index_TipZ,
                    middle_McpX, middle_McpY, middle_McpZ,
                    middle_PipX, middle_PipY, middle_PipZ,
                    middle_DipX, middle_DipY, middle_DipZ,
                    middle_TipX, middle_TipY, middle_TipZ,
                    ring_McpX, ring_McpY, ring_McpZ,
                    ring_PipX, ring_PipY, ring_PipZ,
                    ring_DipX, ring_DipY, ring_DipZ,
                    ring_TipX, ring_TipY, ring_TipZ,
                    pinky_McpX, pinky_McpY, pinky_McpZ,
                    pinky_PipX, pinky_PipY, pinky_PipZ,
                    pinky_DipX, pinky_DipY, pinky_DipZ,
                    pinky_TipX, pinky_TipY, pinky_TipZ,
                    annotated_image)


def load_model():
    num_classes = 26
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1,
                               padding="causal", activation="relu", input_shape=(63, 1)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=256, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(rate=0.2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.load_weights('model/model_SIBI.h5')
    return model


def predict(frame, model):
    (wristX, wristY, wristZ,
     thumb_CmcX, thumb_CmcY, thumb_CmcZ,
     thumb_McpX, thumb_McpY, thumb_McpZ,
     thumb_IpX, thumb_IpY, thumb_IpZ,
     thumb_TipX, thumb_TipY, thumb_TipZ,
     index_McpX, index_McpY, index_McpZ,
     index_PipX, index_PipY, index_PipZ,
     index_DipX, index_DipY, index_DipZ,
     index_TipX, index_TipY, index_TipZ,
     middle_McpX, middle_McpY, middle_McpZ,
     middle_PipX, middle_PipY, middle_PipZ,
     middle_DipX, middle_DipY, middle_DipZ,
     middle_TipX, middle_TipY, middle_TipZ,
     ring_McpX, ring_McpY, ring_McpZ,
     ring_PipX, ring_PipY, ring_PipZ,
     ring_DipX, ring_DipY, ring_DipZ,
     ring_TipX, ring_TipY, ring_TipZ,
     pinky_McpX, pinky_McpY, pinky_McpZ,
     pinky_PipX, pinky_PipY, pinky_PipZ,
     pinky_DipX, pinky_DipY, pinky_DipZ,
     pinky_TipX, pinky_TipY, pinky_TipZ,
     output_IMG) = extract_feature(frame)

    input_IMG = np.array([[[wristX], [wristY], [wristZ],
                           [thumb_CmcX], [thumb_CmcY], [thumb_CmcZ],
                           [thumb_McpX], [thumb_McpY], [thumb_McpZ],
                           [thumb_IpX], [thumb_IpY], [thumb_IpZ],
                           [thumb_TipX], [thumb_TipY], [thumb_TipZ],
                           [index_McpX], [index_McpY], [index_McpZ],
                           [index_PipX], [index_PipY], [index_PipZ],
                           [index_DipX], [index_DipY], [index_DipZ],
                           [index_TipX], [index_TipY], [index_TipZ],
                           [middle_McpX], [middle_McpY], [middle_McpZ],
                           [middle_PipX], [middle_PipY], [middle_PipZ],
                           [middle_DipX], [middle_DipY], [middle_DipZ],
                           [middle_TipX], [middle_TipY], [middle_TipZ],
                           [ring_McpX], [ring_McpY], [ring_McpZ],
                           [ring_PipX], [ring_PipY], [ring_PipZ],
                           [ring_DipX], [ring_DipY], [ring_DipZ],
                           [ring_TipX], [ring_TipY], [ring_TipZ],
                           [pinky_McpX], [pinky_McpY], [pinky_McpZ],
                           [pinky_PipX], [pinky_PipY], [pinky_PipZ],
                           [pinky_DipX], [pinky_DipY], [pinky_DipZ],
                           [pinky_TipX], [pinky_TipY], [pinky_TipZ]]])

    predictions = model.predict(input_IMG)
    char = chr(np.argmax(predictions)+65)
    confidece = np.max(predictions)/np.sum(predictions)
    # put char and confidence on image
    cv2.putText(output_IMG, char, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(output_IMG, str(confidece), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return char, confidece, output_IMG


class Detection(NamedTuple):
    name: str
    prob: float


class VideoTransformer(VideoProcessorBase):

    result_queue: "queue.Queue[List[Detection]]"

    def __init__(self) -> None:
        self.threshold1 = 224
        self.result_queue = queue.Queue()
        self.data = np.ndarray(shape=(1, 240, 240, 3), dtype=np.float32)

    def _predict_image(self, image):
        result: List[Detection] = []
        model = load_model()
        label, confidence, output_img = predict(image, model)
        # st.write([label, confidence])
        result.append(Detection(name=label, prob=float(confidence)))
        return result, output_img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame = frame
        result, output_img = self._predict_image(
            frame.to_ndarray(format="bgr24"))
        self.result_queue.put(result)

        return av.VideoFrame.from_ndarray(output_img, format="bgr24")


def sign_detection(db, user_id):
    st.image("static/sign.jpg")
    ctx = webrtc_streamer(
        key="SpeechSign",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [
            {
                "urls": "stun:openrelay.metered.ca:80",
            },
            {
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443?transport=tcp",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoTransformer,
        async_processing=True,)

    if st.checkbox("Show the detected labels", value=True):
        if ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if ctx.video_processor:
                    try:
                        result = ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                        doc_ref = db.collection(u'users').document(user_id).collection(
                            'sign-detected').add({result[0].name: result[0].prob})
                    except queue.Empty:
                        result = None

                    labels_placeholder.table(result)
                else:
                    break


HERE = Path(__file__).parent


def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def deepspech_model_load(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
    from deepspeech import Model

    model = Model(model_path)
    model.enableExternalScorer(lm_path)
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam)

    stream = model.createStream()

    return model, stream


def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [
            {
                "urls": "stun:openrelay.metered.ca:80",
            },
            {
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443?transport=tcp",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    text = ""

    while True:
        if webrtc_ctx.audio_receiver:
            if stream is None:

                model, stream = deepspech_model_load(
                    model_path, lm_path, lm_alpha, lm_beta, beam)
                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")

                # write to audio_proc/data.txt
                with open('audio_proc/data.txt', 'w') as f:
                    f.write(text)
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


def speech_detection():

    MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    MODEL_LOCAL_PATH = HERE / "model/deepspeech-0.9.3-models.pbmm"
    LANG_MODEL_LOCAL_PATH = HERE / "model/deepspeech-0.9.3-models.scorer"

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH,
                  expected_size=953363776)

    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284
    beam = 100

    app_sst(str(MODEL_LOCAL_PATH), str(
        LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam)

    # read content of audio_proc/data.txt
    with open('audio_proc/data.txt', 'r') as f:
        text = f.read()
        if text != '':
            text = text.upper()
            text = text.replace(' ', '')
            mean_width = 0
            mean_height = 0
            num_of_images = len(text)
            for i in text:
                im = Image.open("static/sign_alpha/"+i+".jpg")
                width, height = im.size
                mean_width += width
                mean_height += height

            mean_width = int(mean_width / num_of_images)
            mean_height = int(mean_height / num_of_images)
            images = []
            for i in text:
                im = Image.open("static/sign_alpha/"+i+".jpg")
                width, height = im.size

                imResize = im.resize(
                    (mean_width, mean_height), Image.Resampling.LANCZOS)
                imResize.save("video_proc/"+i+".jpeg", 'JPEG', quality=95)

            video_name = 'video_proc/{}.webm'.format(text)
            frame = cv2.imread("video_proc/"+text[0]+".jpeg")
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

            for i in text:
                video.write(cv2.imread("video_proc/"+i+".jpeg"))

            cv2.destroyAllWindows()
            video.release()
            st.header(text)
            st.video("video_proc/{}.webm".format(text))

    # clear content of audio_proc/data.txt
    with open('audio_proc/data.txt', 'w') as f:
        f.write('')

    return 0


def show_database(db, user_id):
    doc_ref = db.collection(u'users').document(user_id)
    doc = doc_ref.get()
    user_det = doc.to_dict()
    st.header("User Details")
    user_df = pd.DataFrame({"Type": ['Name', 'DOB', 'Email'], "Value": [
                           user_det['name'], user_det['dob'], user_det['email']]})
    st.dataframe(user_df)

    st.header('Sign Detected')
    sign_df = pd.DataFrame(columns=['Alphabet', 'Confidence'])
    sign_ref = doc_ref.collection(u'sign-detected').stream()
    for sign in sign_ref:
        res = sign.to_dict()
        for alphabet, prob in res.items():
            # use pandas.concat
            sign_df = pd.concat([sign_df, pd.DataFrame(
                {"Alphabet": [alphabet], "Confidence": [prob]})], ignore_index=True)

    st.dataframe(sign_df)


def main():
    image = Image.open("static/vid_call.jpg")
    logo = Image.open("static/logo.png")
    st.set_page_config(page_title="SpeechSign", page_icon=logo)

    st.image(image)
    st.title("@SpeechSign")
    user_id, query_param = cache_query_param()

    app, db = firebase()
    st.sidebar.title("Select the process to your convinience")
    st.sidebar.markdown("Select the conversion method accordingly:")
    algo = st.sidebar.selectbox(
        "Select the Operation", options=["Sign-to-Speech", "Speech-to-Sign", "Access Database", "Sign Recog Model Architecture"]
    )

    if algo == "Sign-to-Speech":
        sign_detection(db, user_id=user_id)
    elif algo == "Speech-to-Sign":
        speech_detection()
    elif algo == "Access Database":
        show_database(db, user_id=user_id)
    elif algo == "Sign Recog Model Architecture":
        st.title("Sign Recog Model Architecture")
        st.image("static/arch.png")


if __name__ == "__main__":
    main()
