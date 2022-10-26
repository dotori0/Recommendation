import numpy as np
from MF import *

def create_item_matrix():

  with open('data/rating.csv', 'r', encoding='utf-8-sig') as f:
    rating = np.genfromtxt(f, delimiter=',')

  with open('data/user_matrix.csv', 'r', encoding='utf-8-sig') as f:
    user = np.genfromtxt(f, delimiter=',')

  item = make_item_matrix(rating, user)
  np.savetxt('data/item_matrix.csv', item, delimiter=',', encoding='utf-8-sig')
  
  print('item matrix renewed')

if __name__ == '__main__':
  create_item_matrix()