#import libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import sklearn
from PIL import Image
#Initialize the flask App
app = Flask(__name__)
cors = CORS(app, resources={r"/api/": {"origins": ""}})
# cors = CORS(app)
 
# CORS Headers
@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response

features_label = np.load('features_label.npy', allow_pickle=True)
features = []
for i in range(0, len(features_label)):
    features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                features_label[i][2], features_label[i][3],
                features_label[i][4]), axis=0))

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit_transform(np.array(features))

randomeforest_model = pickle.load(open('randomforest_model.pkl', 'rb'))



#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app

@app.route('/gender_prediction',methods=['GET','POST'])
def predict_gender():
    if request.method == "POST":
        if 'audio_file' not in request.files:
            return jsonify({'success': True,'data': 'false'})
        
        feat = extract_features(request.files.get('audio_file'))
        features = np.concatenate((feat[0], feat[1], feat[2], feat[3],feat[4]), axis=0)
        c = scaler.transform([features])
        prediction = randomeforest_model.predict(c)
        return jsonify({ 'success': True,'data': '{}'.format(prediction[0])})
            
    return jsonify({'success': True,'data': False})



def extract_features(files):
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(files, res_type='kaiser_fast') 

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)


    return mfccs, chroma, mel, contrast, tonnetz




if __name__ == "__main__":
    app.run(debug=True)
