from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from wtforms.validators import ValidationError
import imghdr


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

class ImageForm(FlaskForm):
    image = FileField('image', validators=[
        FileRequired(message='File is required!'),
        FileAllowed(['jpg', 'png', 'tiff'], 'Images only!')
        ])
    submit = SubmitField('Submit')

    def validate_image(self, image):
        content_type = validate_image(image.data.stream)
        if content_type is None:
            raise ValidationError('File is not a valid image.')
