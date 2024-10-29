from flask import Flask, render_template, request, send_file, jsonify
from io import BytesIO
from PIL import Image
import os
import logging

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = 'encodeImages/'
ENCODED_TEXTS_FOLDER = 'encoded_texts/'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, ENCODED_TEXTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class TextSteganography:
    def __init__(self):
        # Initialize binary to zero-width character mapping
        self.binary_to_zwc = {
            '0': '\u200b',  # Zero-width space
            '1': '\u200c'   # Zero-width non-joiner
        }
        # Create reverse mapping for decoding
        self.zwc_to_binary = {v: k for k, v in self.binary_to_zwc.items()}
    
    def text_to_binary(self, text):
        """Convert text to binary string"""
        binary = ''
        for char in text:
            # Convert each character to 8-bit binary
            binary += format(ord(char), '08b')
        return binary
    
    def binary_to_text(self, binary):
        """Convert binary string back to text"""
        text = ''
        # Process 8 bits at a time
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            # Convert binary to integer and then to character
            text += chr(int(byte, 2))
        return text
    
    def hide_message(self, secret_message, carrier_text):
        """Hide secret message within carrier text using zero-width characters"""
        # Convert secret message to binary
        binary_message = self.text_to_binary(secret_message)
        
        # Insert zero-width characters after each carrier text character
        result = ''
        binary_index = 0
        
        for char in carrier_text:
            result += char
            if binary_index < len(binary_message):
                # Add corresponding zero-width character
                result += self.binary_to_zwc[binary_message[binary_index]]
                binary_index += 1
        
        # If we have remaining bits, append them at the end
        while binary_index < len(binary_message):
            result += self.binary_to_zwc[binary_message[binary_index]]
            binary_index += 1
            
        return result
    
    def extract_message(self, steganographic_text):
        """Extract hidden message from text containing zero-width characters"""
        # Extract zero-width characters and convert to binary
        binary = ''
        for char in steganographic_text:
            if char in self.zwc_to_binary:
                binary += self.zwc_to_binary[char]
        
        # Convert binary back to text
        return self.binary_to_text(binary)

# Create instance of TextSteganography
text_stegno = TextSteganography()

@app.route('/text-encode', methods=['POST'])
def text_encode():
    """Handle text encoding requests"""
    try:
        original_message = request.form.get('originalMessage')
        secret_message = request.form.get('secretMessage')
        
        if not original_message or not secret_message:
            return jsonify({'error': 'Both original and secret messages are required'}), 400

        # Encode the message
        encoded_text = text_stegno.hide_message(secret_message, original_message)
        
        # Generate unique filename
        filename = f"encoded_text_{len(os.listdir(ENCODED_TEXTS_FOLDER))}.txt"
        filepath = os.path.join(ENCODED_TEXTS_FOLDER, filename)
        
        # Save encoded text to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(encoded_text)
        
        return jsonify({
            'success': True,
            'encoded_text': encoded_text,
            'filename': filename
        })

    except Exception as e:
        print(f"Text encode error: {str(e)}")
        return jsonify({'error': 'Failed to encode message'}), 500

@app.route('/text-decode', methods=['POST'])
def text_decode():
    """Handle text decoding requests"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the content of the uploaded file
        content = file.read().decode('utf-8')
        
        # Extract hidden message
        hidden_message = text_stegno.extract_message(content)
        
        return jsonify({
            'success': True,
            'hidden_data': hidden_message
        })

    except Exception as e:
        print(f"Text decode error: {str(e)}")
        return jsonify({'error': 'Failed to decode message'}), 500
    
        
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

@app.route('/textSteganography')
def text_steganography():
    """Render the image steganography page"""
    return render_template('textSteganography.html')


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