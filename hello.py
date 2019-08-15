import os
import joblib
import pandas as pd

from flask import (
    Flask, flash, request, jsonify, abort, redirect, url_for, 
    render_template, send_file)
from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])

app = Flask(__name__)
knn = joblib.load('knn.pkl')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    print('hello_world called 3')
    return 'Hello, my best friend!'

@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s' % username

@app.route('/square/<int:number>')
def square(number):
    sq = number * number
    return f'Number: {number}. Square: {sq}'

@app.route('/avg/<nums>')
def avg(nums):
    parts = nums.split(',')
    numbers = [float(p) for p in parts]
    mean = sum(numbers) / len(numbers)
    return f'<p>Numbers: {numbers}</p><p>Mean: {mean:.4f}</p>'

@app.route('/irisdemo')
def irisdemo():
    import numpy as np
    from sklearn import datasets
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target

    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_X_test = iris_X[indices[-10:]]
    iris_y_test = iris_y[indices[-10:]]

    # Create and fit a nearest-neighbor classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)

    return str(knn.predict(iris_X_test))

@app.route('/iris/<param>')
def iris(param):
    parts = param.split(',')
    numbers = [float(p) for p in parts]

    result = knn.predict([numbers])
    return str(result)

@app.route('/showimage')
def show_image():
    setosa_url = 'setosa.jpg'
    return f'<img src="static/{setosa_url}" alt="setosa"/>'

@app.route('/badrequest400')
def bad_request():
    return abort(400)

@app.route('/iris_post', methods=['POST'])
def iris_post():
    content = request.get_json()
    
    try:
        parts = content['flower'].split(',')
        numbers = [float(p) for p in parts]
    except:
        return redirect(url_for('bad_request'))

    result = knn.predict([numbers])
    
    return jsonify({'class': int(result[0])})

    # return str(result[0])
    #return jsonify(content)

app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key",
))

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    file = FileField()

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    form = MyForm()
    if form.validate_on_submit():
        #return redirect('/success')
        return f"Form submitted. Name: {form.name.data}"
    return render_template('submit.html', form=form)

@app.route('/submitfile', methods=['GET', 'POST'])
def submitfile():
    form = MyForm()
    if form.validate_on_submit():
        filedata = form.file.data
        filename = secure_filename(form.name.data)
        # filedata.save(os.path.join(
        #     'uploads', filename
        # ))

        df = pd.read_csv(filedata, header=None)
        pred = pd.DataFrame(knn.predict(df))
        filepath = f'results/{filename}.csv'
        pred.to_csv(filepath, index=False)

        return send_file(
            filepath,
            mimetype='text/csv',
            attachment_filename=f'{filename}.csv',
            as_attachment=True)

        #return 'done'

        #return f"Form submitted. Name: {form.name.data}"
    return render_template('submit_file.html', form=form)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))
            return "File uploaded"
    
    return '''
        <!doct[ype html>
        <title>Upload new file</title>
        <h1>Upload file</h1>
        <form method=post enctype=multipart/form-data>
            <input type=file name=file />
            <input type=submit value=Upload />
        </form>
    '''