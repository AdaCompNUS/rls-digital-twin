#!/usr/bin/python3

from google.cloud import texttospeech
import os
import rospy

from rls_control_msgs.srv import *

class TextToSpeechService():
    def __init__(self):
        self._client = texttospeech.TextToSpeechClient()
        self._service = rospy.Service(
            "rls_control_services/text_to_speech", TextToSpeech, self._handle_req)
        rospy.loginfo("text_to_speech service inited")

    def _handle_req(self, req):
        res = TextToSpeechResponse()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=req.text)

        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")

        gender = req.gender
        if gender == 'male':
            voice = texttospeech.VoiceSelectionParams(
                language_code='en-US',
                ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        elif gender == 'female':
            voice = texttospeech.VoiceSelectionParams(
                language_code='en-US',
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        else:
            voice = texttospeech.VoiceSelectionParams(
                language_code='en-US',
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = self._client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'output.wav')
        # The response's audio_content is binary.
        with open(filename, 'wb') as out:
            # Write the response to the output file.
            out.write(response.audio_content)
            print('Audio content written to file {}'.format(filename))
            # os.system("scp output.wav movo@movo2:/home/movo/")
            # os.system("ssh -f movo@movo2 'aplay output.wav &>/dev/null'")
            res.success = True
            res.filename = filename
        return res

if __name__ == "__main__":
    rospy.init_node("text_to_speech_service")
    movo_speaker = TextToSpeechService()
    rospy.spin()
