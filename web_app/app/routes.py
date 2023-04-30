from app import app
from flask import render_template, request, redirect, url_for, flash
from app.forms import ImageForm
from werkzeug.utils import secure_filename
import os




@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = ImageForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            uploaded_file = form.image.data
            filename = secure_filename(uploaded_file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(save_path)
            return redirect(url_for('index'))
        else:
            print(form.errors)
    return render_template('index.html', title='Home', form=form)
