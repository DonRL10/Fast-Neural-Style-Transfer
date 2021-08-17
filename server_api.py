import base64
from flask import Flask, json, request, render_template, jsonify
from flask_cors import CORS

import cv2
import numpy as np

from fst import stylize

app = Flask(__name__, template_folder='templates', static_folder='templates/assets')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/main', methods = ['POST'])
def main():
    #r = request.form.to_dict(flat = False)
    #img_Data = r['image'][0]
    r = request.get_json(force=True)
    img_Data = r['image']
    buffer = np.fromstring(base64.b64decode(img_Data), dtype = np.uint8)
    img = cv2.imdecode(buffer, 1)
    out = stylize(img, 'models/mosaic2.pt', 'output.jpg')
    return jsonify(response = 'data:image/jpeg;base64,'+str(base64.b64encode(out))[2:-1])
    
if __name__ == '__main__':
    app.run(host = 'localhost', threaded = True, debug = True)

