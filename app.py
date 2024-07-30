from flask import Flask, render_template, request, jsonify
import cv2
import base64
import io
from PIL import Image
import numpy as np
from calculate import process_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            type = request.form.get('equationType')
            result, exp = process_image(image, type)
            return jsonify({'result': result, 'exp': exp})
        except Exception as e:
            return jsonify({'error': str(e)})

    elif 'image' in request.form:
        try:
            image_data = request.form['image']
            type = request.form.get('equationType')
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img_array = np.array(img)
            result, exp = process_image(img_array, type)
            return jsonify({'result': result, 'exp': exp})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'No file part'})

if __name__ == '__main__':
    app.run(debug=True)
