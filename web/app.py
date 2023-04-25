from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
# Telling our app where our database is located
# 3/ is relative path
# 4/ is absolute path
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
# Initialize the database with app's config
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False) # Nullable false == we don't want this field to be left blank
    completed = db.Column(db.Integer, default=0)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        new_task = Todo(content=task_content)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/')
        except:
            return "There was an issue adding your task."

    else:
        # look at the database and retun all in a certin order
        tasks = Todo.query.order_by(Todo.date_created).all()
        # Show the page
        return render_template("index.html", tasks=tasks)
    

if __name__ == "__main__":
    # set up debug mode, so all errors show up in the web app
    app.run(debug=True)