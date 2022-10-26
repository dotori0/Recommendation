import os
import numpy as np
import csv
from MF import *
from create_item_matrix import *

from flask import Flask, jsonify, render_template

create_item_matrix()

f = open('data/item_index.csv', 'rt', encoding="cp949")
index = next(csv.reader(f))

with open('data/item_matrix.csv', 'r', encoding='utf-8-sig') as f:
  item_matrix = np.genfromtxt(f, delimiter=',')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def form():
  return render_template('index.html')

@app.route('/<user_vector>')
def result(user_vector):
  user_vector = list(user_vector)
  user_vector = [float(i) for i in user_vector]
  user_vector = np.array(user_vector)

  subject = recommendation(user_vector, item_matrix, index)
  return render_template('result.html', subject=subject)

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=False)