from flask import render_template
from app import app



@app.errorhandler(400)
def file_not_compatible(error):
    return render_template('400.html'), 400

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(413)
def file_too_big(error):
    return render_template('413.html'), 413

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500