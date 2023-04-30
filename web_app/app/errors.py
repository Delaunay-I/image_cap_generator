from flask import render_template
from app import app


@app.errorhandler(413)
def file_too_big(error):
    return render_template('413.html')

@app.errorhandler(400)
def file_not_compatible(error):
    return render_template('400.html')