import pocketsphinx as ps
from pocketsphinx import Decoder

# Create a decoder with certain model
config = ps.Config()
config.set_string('-hmm', 'en-us')
config.set_string('-lm', 'en-us.lm.bin')
config.set_string('-dict', 'cmudict-en-us.dict')
config.set_string('-allphone', 'en-us-phone.lm.bin')
config.set_string('-lw', 2.0)
config.set_string('-beam', 1e-20)
config.set_string('-pbeam', 1e-20)
config.set_string('-wbeam', 1e-20)
config.set_string('-pip', 0.3)	
config.set_string('-mmap', False)
decoder = Decoder(config)

f = open("transcript.wav", "rb")
decoder.start_utt()
while True:
    buf = f.read(1024)
    if buf:
        decoder.process_raw(buf, False, False)
    else:
        break
decoder.end_utt()

for seg in decoder.seg():
    print(seg.word, seg.prob, seg.start_frame, seg.end_frame)

f.close()    



