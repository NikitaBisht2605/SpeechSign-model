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


# np.set_printoptions(suppress=True)


global query_param


def firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            'static/speechsign-23477-8f5b84f0980a.json')
        app = firebase_admin.initialize_app(cred)
    app = firebase_admin.get_app()
    db = firestore.client()
    return app, db


@st.cache
def cache_query_param():
    query_param = st.experimental_get_query_params()
    print(query_param)
    user_id = query_param['user'][0]
    return user_id, query_param


def gen_labels():
    labels = {}
    with open("model/labels.txt", "r") as label:
        text = label.read()
        lines = text.split("\n")
        for line in lines[0:-1]:
            hold = line.split(" ", 1)
            labels[hold[0]] = hold[1]
    return labels


class Detection(NamedTuple):
    name: str
    prob: float


class VideoTransformer(VideoProcessorBase):

    result_queue: "queue.Queue[List[Detection]]"

    def __init__(self) -> None:
        self.threshold1 = 224
        self.result_queue = queue.Queue()
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    def _predict_image(self, image, model):
        result: List[Detection] = []
        labels = gen_labels()
        prediction = model.predict(image)
        confidence = max(prediction[0])
        st.write(confidence)
        idx = np.where(prediction[0] == confidence)
        alpha = labels.get(str(idx[0][0]))
        result.append(Detection(name=alpha, prob=float(confidence)))
        return result

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame = frame
        img = frame.to_ndarray(format="bgr24")
        frm = cv2.resize(img, (224, 224))
        frm = Image.fromarray(frm)
        size = (224, 224)
        image = ImageOps.fit(frm, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array
        model = keras.models.load_model("model/keras_model.h5", compile=False)
        result = self._predict_image(self.data, model)
        self.result_queue.put(result)

        return 0


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
                from deepspeech import Model

                model = Model(model_path)
                model.enableExternalScorer(lm_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

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


def app_sst_with_video(
    model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int
):
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
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
        media_stream_constraints={"video": True, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.state.playing:
            if stream is None:
                from deepspeech import Model

                model = Model(model_path)
                model.enableExternalScorer(lm_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()

            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
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
                with open('audio_proc/data.txt', 'w') as f:
                    f.write(text)
        else:
            status_indicator.write("Stopped.")
            break


def speech_detection():

    MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
    LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH,
                  expected_size=953363776)

    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284
    beam = 100

    sound_only_page = "Sound only (sendonly)"
    with_video_page = "With video (sendrecv)"

    app_mode = st.selectbox(
        "Choose the app mode",
        [sound_only_page, with_video_page],
        index=1,
    )

    if app_mode == sound_only_page:
        app_sst(
            str(MODEL_LOCAL_PATH), str(
                LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
        )
    elif app_mode == with_video_page:
        app_sst_with_video(
            str(MODEL_LOCAL_PATH), str(
                LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
        )

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
