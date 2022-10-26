import numpy as np


# 의사 역행렬
def pseudo_inverse(matrix):
  u, Sigma, vt = np.linalg.svd(matrix, full_matrices=False)
  Sigma = np.diag(Sigma)
  ut = u.T
  v = vt.T
  inv_sigma = np.linalg.inv(Sigma)
  return np.dot(v, np.dot(inv_sigma, ut))


# 의사 역행렬 기반 아이템 행렬
def make_item_matrix(rating, user_matrix):
  inverse_user_matrix = pseudo_inverse(user_matrix)
  return np.dot(inverse_user_matrix, rating)


# 추천
def recommendation(user_vector, item_matrix, index):
  user_vector = np.reshape(user_vector, (1, -1))
  
  recommendation_vector = np.dot(user_vector, item_matrix)
  recommendation_vector = np.reshape(recommendation_vector, (-1,))

  tu = [(index[i], recommendation_vector[i]) 
    for i in range(min(len(index), len(recommendation_vector)))]

  tu.sort(key=lambda x: -x[1])

  return [tu[i][0] for i in range(len(tu))]