from flask import Flask, render_template, request, send_file, jsonify
from io import BytesIO
from PIL import Image
import os
import logging

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'encodeImages/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class IMG_Stegno:
    def generate_Data(self, data):
        """Convert encoding data into 8-bit binary form"""
        new_data = []
        for i in data:
            new_data.append(format(ord(i), '08b'))
        return new_data

    def modify_Pix(self, pix, data):
        """Modify pixels to encode data"""
        dataList = self.generate_Data(data)
        dataLen = len(dataList)
        imgData = iter(pix)
        for i in range(dataLen):
            pix = [value for value in imgData.__next__()[:3] +
                   imgData.__next__()[:3] +
                   imgData.__next__()[:3]]
            for j in range(0, 8):
                if (dataList[i][j] == '0') and (pix[j] % 2 != 0):
                    pix[j] -= 1
                elif (dataList[i][j] == '1') and (pix[j] % 2 == 0):
                    pix[j] -= 1
            if (i == dataLen - 1):
                if (pix[-1] % 2 == 0):
                    pix[-1] -= 1
            else:
                if (pix[-1] % 2 != 0):
                    pix[-1] -= 1
            pix = tuple(pix)
            yield pix[0:3]
            yield pix[3:6]
            yield pix[6:9]

    def encode_enc(self, newImg, data):
        """Encode the data into the image"""
        w = newImg.size[0]
        (x, y) = (0, 0)
        for pixel in self.modify_Pix(newImg.getdata(), data):
            newImg.putpixel((x, y), pixel)
            if (x == w - 1):
                x = 0
                y += 1
            else:
                x += 1

    def decode(self, image):
        """Decode the data from the image"""
        image_data = iter(image.getdata())
        data = ''
        while (True):
            pixels = [value for value in image_data.__next__()[:3] +
                      image_data.__next__()[:3] +
                      image_data.__next__()[:3]]
            binary_str = ''
            for i in pixels[:8]:
                if i % 2 == 0:
                    binary_str += '0'
                else:
                    binary_str += '1'
            data += chr(int(binary_str, 2))
            if pixels[-1] % 2 != 0:
                return data

# Routes
@app.route('/')
def index():
    """Render the main page with steganography method selection"""
    return render_template('index.html')

@app.route('/imageSteganography')
def image_steganography():
    """Render the image steganography page"""
    return render_template('imageSteganography.html')

@app.route('/encode', methods=['POST'])
def encode():
    """Handle image encoding requests"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image = request.files['image']
        data = request.form.get('data')
        
        if not data:
            return jsonify({'error': 'No data provided for encoding'}), 400
        
        if not image.filename:
            return jsonify({'error': 'No image selected'}), 400

        # Process the image
        img = Image.open(image)
        new_img = img.copy()
        stegno = IMG_Stegno()
        
        try:
            stegno.encode_enc(new_img, data)
        except Exception as e:
            print(f"Encoding error: {str(e)}")
            return jsonify({'error': 'Failed to encode data into image'}), 500

        # Save and return the encoded image
        output_path = os.path.join(UPLOAD_FOLDER, f"{os.path.splitext(image.filename)[0]}_encoded.png")
        new_img.save(output_path, format='PNG')
        
        return send_file(
            output_path,
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{os.path.splitext(image.filename)[0]}_encoded.png"
        )

    except Exception as e:
        print(f"Encode route error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/decode', methods=['POST'])
def decode():
    """Handle image decoding requests"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image = request.files['image']
        
        if not image.filename:
            return jsonify({'error': 'No image selected'}), 400

        # Process the image
        img = Image.open(image)
        stegno = IMG_Stegno()
        
        try:
            hidden_data = stegno.decode(img)
            return jsonify({'hidden_data': hidden_data})
        except Exception as e:
            print(f"Decoding error: {str(e)}")
            return jsonify({'error': 'Failed to decode data from image'}), 500

    except Exception as e:
        print(f"Decode route error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)