from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from wtforms.validators import ValidationError


def validate_image_content(stream):
    # Read the first 4 bytes of the file
    header = stream.read(4)
    # Reset the stream position
    stream.seek(0)
    # Convert the bytes to hex format
    hex_header = header.hex().upper()
    # Check if the hex header matches any of the image formats
    if hex_header.startswith('FFD8FF'):
        return '.jpg'
    elif hex_header.startswith('89504E47'):
        return '.png'
    else:
        return None

class ImageForm(FlaskForm):
    image = FileField('image', validators=[
        FileRequired(message='File is required!'),
        FileAllowed(['jpg', 'png'], 'Images only!')
        ])
    submit = SubmitField('Submit')

    def validate_image(self, image):
        content_type = validate_image_content(image.data.stream)
        if content_type is None:
            raise ValidationError('File is not a valid image.')
        # Add this check for file extension
        if content_type not in ['.jpg', '.png']:
            raise ValidationError('Images only!')
