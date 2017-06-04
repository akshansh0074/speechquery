#!/usr/bin/python

from __future__ import division

import contextlib
import threading
import pyaudio

from gcloud.credentials import get_credentials
from google.cloud.speech.v1beta1 import cloud_speech_pb2 as cloud_speech
from google.rpc import code_pb2
from grpc.beta import implementations

import rospy
from google_cloud_speech.msg import ResultTranscript
from std_msgs.msg import Empty

# Audio recording parameters
RATE = 16000
CHANNELS = 1
CHUNK = int(RATE / 10)  # 100ms

# Keep the request alive for this many seconds
DEADLINE_SECS = 8 * 60 * 60
SPEECH_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'


class GoogleSpeech():
    def __init__(self):

        self.pub_transcript = rospy.Publisher('result_transcript', ResultTranscript, queue_size=5)
        self.pub_start_speech = rospy.Publisher('start_of_speech', Empty, queue_size=10)
        self.pub_end_speech = rospy.Publisher('end_of_speech', Empty, queue_size=10)

        self.is_start_audio = False
        self.is_start_speech = False
        self.is_stop_audio = True

        self.stop_audio = threading.Event()
        with cloud_speech.beta_create_Speech_stub(self.make_channel('speech.googleapis.com', 443)) as self.service:
            self.t1 = threading.Thread(target=self.listen_print_loop)
            self.t1.start()

    def request_stream(self, stop_audio, channels=CHANNELS, rate=RATE, chunk=CHUNK):
        recognition_config = cloud_speech.RecognitionConfig(
            encoding='LINEAR16',
            sample_rate=rate,
            language_code='ko-KR',
        )
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
            single_utterance=False
        )

        yield cloud_speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        with self.record_audio(channels, rate, chunk) as audio_stream:
            while not rospy.is_shutdown():
                data = audio_stream.read(chunk)
                if not data:
                    raise StopIteration()
                yield cloud_speech.StreamingRecognizeRequest(audio_content=data)

    @contextlib.contextmanager
    def record_audio(self, channels, rate, chunk):
        audio_interface = pyaudio.PyAudio()
        audio_stream = audio_interface.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                                            frames_per_buffer=chunk, )
        yield audio_stream

        audio_stream.stop_stream()
        audio_stream.close()
        audio_interface.terminate()

    def listen_print_loop(self):
        recognize_stream = self.service.StreamingRecognize(self.request_stream(self.stop_audio), DEADLINE_SECS)

        for resp in recognize_stream:
            if resp.error.code != code_pb2.OK:
                raise RuntimeError('Server error: ' + resp.error.message)

            if self.is_stop_audio:
                if resp.endpointer_type == 1:
                    self.is_start_audio = True
                elif resp.endpointer_type == 2:
                    self.is_start_audio = False
                    self.is_stop_audio = True

                if resp.endpointer_type == 0 and self.is_start_audio:
                    if not self.is_start_speech:
                        self.is_start_speech = True
                        self.pub_start_speech.publish()

                elif resp.endpointer_type == 0 and self.is_stop_audio:
                    if self.is_start_speech:
                        self.is_start_speech = False
                        self.pub_end_speech.publish()

            for result in resp.results:
                if result.is_final:
                    msg = ResultTranscript()
                    msg.transcript = result.alternatives[0].transcript
                    msg.confidence = result.alternatives[0].confidence
                    self.pub_transcript.publish(msg)

    def make_channel(self, host, port):
        ssl_channel = implementations.ssl_channel_credentials(None, None, None)
        creds = get_credentials().create_scoped([SPEECH_SCOPE])

        auth_header = ('Authorization', 'Bearer ' + creds.get_access_token().access_token)
        auth_plugin = implementations.metadata_call_credentials(
            lambda _, cb: cb([auth_header], None),
            name='google_creds')

        composite_channel = implementations.composite_channel_credentials(ssl_channel, auth_plugin)
        return implementations.secure_channel(host, port, composite_channel)


if __name__ == '__main__':
    rospy.init_node('google_speech', anonymous=False)
    speech = GoogleSpeech()
    rospy.spin()