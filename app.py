import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './upload'
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/', methods=['GET', 'POST'])
def index():
    original_image_path = None
    transformed_image_path = None
    if request.method == 'POST':
        # Handle image upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(file.filename)
            image_id = f"image_{len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1}"
            image_dir = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
            os.makedirs(image_dir, exist_ok=True)
            original_image_path = os.path.join(image_dir, f"{filename}_original.jpg")
            file.save(original_image_path)
            transformed_image_path = os.path.join(image_dir, f"{filename}_transformed.jpg")

            # Store image_id and filename in session
            session['image_id'] = image_id
            session['filename'] = filename

            # Set initial transformation for live preview
            brightness = float(request.form.get('brightness', 1.0))
            contrast = float(request.form.get('contrast', 1.0))
            sharpness = float(request.form.get('sharpness', 1.0))
            sepia = float(request.form.get('sepia', 0))
            blur = int(request.form.get('blur', 0))
            edge_detection = int(request.form.get('edge-detection', 50))

            apply_transformations(original_image_path, transformed_image_path, brightness, contrast, sharpness, sepia,
                                  blur, edge_detection)

            # Store the transformed image path and transformation parameters in session
            session['transformed_image_path'] = transformed_image_path
            session['brightness'] = brightness
            session['contrast'] = contrast
            session['sharpness'] = sharpness
            session['sepia'] = sepia
            session['blur'] = blur
            session['edge_detection'] = edge_detection

            return redirect(url_for('index'))
        elif 'transform' in request.form and 'image_id' in session:
            # Handle transformation of already uploaded image for live preview
            image_id = session['image_id']
            filename = session['filename']
            image_dir = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
            original_image_path = os.path.join(image_dir, f"{filename}_original.jpg")
            transformed_image_path = os.path.join(image_dir, f"{filename}_transformed.jpg")

            # Set transformation parameters
            brightness = float(request.form.get('brightness', session.get('brightness', 1.0)))
            contrast = float(request.form.get('contrast', session.get('contrast', 1.0)))
            sharpness = float(request.form.get('sharpness', session.get('sharpness', 1.0)))
            sepia = float(request.form.get('sepia', session.get('sepia', 0)))
            blur = int(request.form.get('blur', session.get('blur', 0)))
            edge_detection = int(request.form.get('edge-detection', session.get('edge_detection', 50)))

            apply_transformations(original_image_path, transformed_image_path, brightness, contrast, sharpness, sepia,
                                  blur, edge_detection)

            # Store the latest parameters in session
            session['brightness'] = brightness
            session['contrast'] = contrast
            session['sharpness'] = sharpness
            session['sepia'] = sepia
            session['blur'] = blur
            session['edge_detection'] = edge_detection
            session['transformed_image_path'] = transformed_image_path

            return redirect(url_for('index'))
        elif 'save' in request.form and 'image_id' in session:
            # Save the transformed image and parameters
            image_id = session['image_id']
            filename = session['filename']
            image_dir = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
            transformed_image_path = os.path.join(image_dir, f"{filename}_transformed.jpg")
            json_path = os.path.join(image_dir, f"{filename}.json")

            # Get the current transformation parameters
            brightness = session.get('brightness', 1.0)
            contrast = session.get('contrast', 1.0)
            sharpness = session.get('sharpness', 1.0)
            sepia = session.get('sepia', 0)
            blur = session.get('blur', 0)
            edge_detection = session.get('edge_detection', 50)

            # Save transformation parameters
            parameters = {'brightness': brightness, 'contrast': contrast, 'sharpness': sharpness, 'sepia': sepia,
                          'blur': blur, 'edge_detection': edge_detection}
            with open(json_path, 'w') as f:
                json.dump(parameters, f)

            flash('Image and parameters saved successfully!', 'success')
            return redirect(url_for('index'))
    else:
        # Display transformed image after transformation
        if 'image_id' in session:
            image_id = session['image_id']
            filename = session['filename']
            original_image_path = url_for('uploaded_file', image_id=image_id, filename=f"{filename}_original.jpg")
            transformed_image_path = url_for('uploaded_file', image_id=image_id, filename=f"{filename}_transformed.jpg")
        elif 'transformed_image_path' in session:
            transformed_image_path = session['transformed_image_path']

    # Get the current transformation parameters from session
    brightness = session.get('brightness', 1.0)
    contrast = session.get('contrast', 1.0)
    sharpness = session.get('sharpness', 1.0)
    sepia = session.get('sepia', 0)
    blur = session.get('blur', 0)
    edge_detection = session.get('edge_detection', 50)

    return render_template('index.html', original=original_image_path, transformed=transformed_image_path,
                           filename=session.get('filename'), brightness=brightness, contrast=contrast,
                           sharpness=sharpness, sepia=sepia, blur=blur, edge_detection=edge_detection)


@app.route('/upload/<image_id>/<filename>')
def uploaded_file(image_id, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], image_id), filename)


def apply_transformations(original_path, output_path, brightness, contrast, sharpness, sepia, blur, edge_detection):
    image = cv2.imread(original_path)
    if image is None:
        return

    image = apply_brightness(image, brightness)
    image = apply_contrast(image, contrast)
    image = apply_sharpness(image, sharpness)
    image = apply_sepia(image, sepia)
    image = apply_blur(image, blur)
    image = apply_edge_detection(image, edge_detection)

    cv2.imwrite(output_path, image)


def apply_brightness(image, brightness):
    return cv2.convertScaleAbs(image, alpha=brightness, beta=0)


def apply_contrast(image, contrast):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_sharpness(image, sharpness):
    kernel = np.array([[0, -1, 0], [-1, 5 + sharpness, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def apply_sepia(image, sepia):
    if sepia == 0:
        return image
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, kernel)
    sepia_image = np.clip(sepia_image, 0, 255).astype(image.dtype)
    return cv2.addWeighted(image, 1 - sepia, sepia_image, sepia, 0)


def apply_blur(image, blur):
    if blur == 0:
        return image
    return cv2.GaussianBlur(image, (2 * blur + 1, 2 * blur + 1), 0)


def apply_edge_detection(image, edge_detection):
    if edge_detection == 0:
        return image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, edge_detection, edge_detection * 3)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


if __name__ == '__main__':
    app.run(debug=True)
