from app import app
from flask import render_template, redirect, url_for, send_from_directory, flash
from app.forms import ImageForm
from werkzeug.utils import secure_filename
import os
import tensorflow as tf


model_name = "base_imgCap_model.h5"
model_path = os.path.join(app.config['BASEDIR'], 'tf_files', model_name)
model = tf.keras.models.load_model(model_path)

from app.tf_predict import predict_beam_search



@app.route('/')
@app.route('/index')
def index():
    form = ImageForm()
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    image_file = []
    caption = None
    print(files, type(files))
    if files:
        files.sort(key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True)
        image_file = files[0] 
        print(image_file)       
    
    return render_template('index.html',
     image_file=image_file, form=form, caption=caption)

@app.route('/', methods=['POST'])
@app.route('/index', methods=[ 'POST'])
def upload_files():
    form = ImageForm()
    if form.validate_on_submit():
        uploaded_file = form.image.data
        filename = secure_filename(uploaded_file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(save_path)
        flash("Image uploaded Successfully!")
        caption = predict_beam_search(save_path, 10, model)

        return render_template('index.html', image_file=filename, caption=caption, form=form)
    else:
        print(form.errors)
    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
