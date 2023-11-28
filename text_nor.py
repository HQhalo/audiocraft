SPECIAL_TERM = {
    '\"4 on the floor\"': "four-on-the-floor", 
    "this is a music piece.": "", 
    "the recording features a": "","the song is an":"",
    "the recording features an": "",
    "song features": "",
    "the low quality recording features a": "",
    "this music is an instrumental.": "",
    "a live performance": "",
    "this is a classical music piece. " : "",
    "this song features": "",
    "there are no voices in this song": "",
    "this song can be": "",
    "It sounds": "",
    "features a" : "",
    "features an": "",
    "this music is": "",
    "this is a": "",
    "this song contains": "",
    "this instrumental": ""
    }

def text_nor(text):
    text = text.lower()
    for term in SPECIAL_TERM:
        text = text.replace(term, SPECIAL_TERM[term])
    return text
# print(text_nor("The Techno song features a repetitive synth bass, groovy piano melody, wooden percussion and punchy \"4 on the floor\" kick pattern. It sounds upbeat and energetic - like something you would hear in clubs."))