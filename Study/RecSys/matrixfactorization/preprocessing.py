import numpy as np
import pandas as pd

np.random.seed(42) # 재현성을 위한 시드 고정

num_users_large = 100
num_items_large = 50
sparsity = 0.05 # 5%의 데이터만 평점이 존재

# R_large 행렬을 0으로 초기화
R_large = np.zeros((num_users_large, num_items_large))

# 평점이 있는 데이터의 개수
num_ratings = int(num_users_large * num_items_large * sparsity)

# 평점을 무작위 위치에 생성 (1~5점)
# row, col 인덱스를 무작위로 뽑음
rows = np.random.randint(0, num_users_large, size=num_ratings)
cols = np.random.randint(0, num_items_large, size=num_ratings)
ratings = np.random.randint(1, 6, size=num_ratings)

# 중복 위치에 평점이 찍히는 것을 방지하기 위해 zip 사용
for r, c, val in zip(rows, cols, ratings):
    R_large[r, c] = val

print(f"생성된 행렬 크기: {R_large.shape}")
print(f"총 원소 개수: {R_large.size}")
print(f"0이 아닌 평점 개수: {np.count_nonzero(R_large)}")
