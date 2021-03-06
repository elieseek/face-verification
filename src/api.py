import pickle
import cv2
import numpy as np
import torch
import flask
from flask import request, jsonify
from faceverification import networks, evaluate

app = flask.Flask(__name__)

@app.route('/', methods=['get'])
def home():
  return 'API running'

@app.route('/embed', methods=['post'])
def embed():
  req = request.files.get('image').read()
  np_img = np.fromstring(req, np.uint8)
  img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
  embedding = evaluate.calc_embedding(model, img)
  return jsonify(embedding.tolist())

if __name__ == '__main__':
  model = networks.ConvEmbedder()
  model.load_state_dict(torch.load('embedder_epoch_93.pt', map_location=torch.device('cpu')))
  with open('mean.pkl', 'rb') as f: mean = pickle.load(f)
  with open('sd.pkl', 'rb') as f: sd = pickle.load(f)
  app.run()