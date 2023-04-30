from app import app
from flask import render_template, request, redirect, url_for, send_from_directory
from app.forms import ImageForm
from werkzeug.utils import secure_filename
import os


@app.route('/')
@app.route('/index')
def index():
    form = ImageForm()
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files.sort(key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True)
    return render_template('index.html', files=files, most_recent_file=files[0], form=form)

@app.route('/', methods=['POST'])
@app.route('/index', methods=[ 'POST'])
def upload_files():
    form = ImageForm()
    if form.validate_on_submit():
        uploaded_file = form.image.data
        filename = secure_filename(uploaded_file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(save_path)
        return redirect(url_for('index'))
    else:
        print(form.errors)
    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)