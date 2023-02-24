import moviepy.editor as mp
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

def speech_transcription(audio_path):

    # Load Wav2Vec2 model and tokenizer
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Load audio file and process it in segments
    framerate = 16000
    input_audio, _ = librosa.load(audio_path, sr=framerate)

    # Use a loop to process the audio file in segments, if it's too large to fit in memory
    segment_length = 30  # in seconds
    num_segments = int(np.ceil(len(input_audio) / (segment_length * framerate)))
    transcription = ''
    
    for i in range(num_segments):
        start = i * segment_length * framerate
        end = (i + 1) * segment_length * framerate
        input_segment = input_audio[start:end]
        input_values = tokenizer(input_segment, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        segment_transcription = tokenizer.batch_decode(predicted_ids)[0]
        transcription += segment_transcription

    return transcription
