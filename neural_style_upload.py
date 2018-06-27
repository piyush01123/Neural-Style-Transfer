import os
os.chdir('/home/ec2-user/neural-style')

import urllib2
import cStringIO
from scipy import ndimage
from skimage.color import rgb2gray
import scipy.misc
from flask import Flask, request
import boto3
from datetime import datetime as dt
import string
from PIL import Image
import json
import neural_style_wrapper

app = Flask(__name__)

@app.route('/')
def index():
    return '''<form method="POST" action="upload">
    <input type="text" name=myfile>
    <input type=submit>
    </form>'''

@app.route('/upload', methods=['POST'])
def upload():
    s3 = boto3.resource('s3', aws_access_key_id='XXXXXXXXXXXXXXXXX', aws_secret_access_key='XXXXXXXXXXXXXXXXXXX')
    url = request.form['myfile']
    print request
    print url
    file = cStringIO.StringIO(urllib2.urlopen(url).read())
    img = ndimage.imread(file, mode = 'RGB')
    scipy.misc.imsave('image_from_url.jpg', img)
    content_file = 'image_from_url.jpg'
    style_file = 'van_gogh.jpg'
    outfile = string.replace('outfile_'+  '_'.join(str(dt.now()).split()) + '.jpg', ':', '-')
    neural_style_wrapper.stylize(content_file, style_file, outfile)
    file = open(outfile, "r")
    s3.Bucket('bucketforflask').put_object(Key=outfile, Body=file)
    #return '<h1>File saved to S3. You are awesome and <a href = "https://s3.amazonaws.com/bucketforflask/'+ outfile + '">here</a> is your converted image</h1>'
    outurl= 'https://s3.amazonaws.com/bucketforflask/'+ outfile
    print outurl
    return json.dumps({'name':outurl}), 200, {'ContentType':'application/json'} 

app.run(host = '0.0.0.0', port = 80)


