import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(42)

def movingAverages1(X,n,k):
    """
    X: array
    n: size of X
    k: k-th order moving average array A of array X
    """
    A = np.zeros(n)
    for i in range(k-1,n):
        A[i] = sum(X[i-k+1:i+1]) / k
    return A

def movingAverages2(X,n,k):
    A = np.zeros(n)
    if n < k:
        return A
    
    current_sum = sum(X[:k])
    A[k-1] = current_sum / k

    for i in range(k,n):
        current_sum = current_sum - X[i-k] + X[i]
        A[i] = current_sum / k
    return A

def problem_1a():
    n = 2**5
    k = 2**4
    ratios = []

    for _ in range(1000):
        X = np.random.uniform(0,1,n)

        start1 = time.perf_counter()
        movingAverages1(X,n,k)
        time1 = time.perf_counter() - start1
        
        start2 = time.perf_counter()
        movingAverages2(X,n,k)
        time2 = time.perf_counter() - start2

        time2 = max(time2, 1e-9)
        ratios.append(time1 / time2)

        plt.figure(figsize=(8, 6))
        plt.hist(ratios, bins=50, color='skyblue', edgecolor='black')
        plt.title('Execution Time Ratio (movingAverages1 / movingAverages2)')
        plt.xlabel('Ratio (Time1 / Time2)')
        plt.ylabel('Frequency')
        plt.savefig('./ratio_hist2.jpg')
        plt.close()

def problem_1b():
    n_list = [2**4, 2**5, 2**6, 2**7, 2**8]
    k = 2**3
    
    min_ratios = []
    max_ratios = []
    avg_ratios = []

    for n in n_list:
        ratios = []
        for _ in range(1000):
            X = np.random.uniform(0, 1, n)
            
            start1 = time.perf_counter()
            movingAverages1(X, n, k)
            time1 = time.perf_counter() - start1
            
            start2 = time.perf_counter()
            movingAverages2(X, n, k)
            time2 = time.perf_counter() - start2
            
            time2 = max(time2, 1e-9)
            ratios.append(time1 / time2)
            
        min_ratios.append(np.min(ratios))
        max_ratios.append(np.max(ratios))
        avg_ratios.append(np.mean(ratios))

    # 선 그래프 저장
    plt.figure(figsize=(8, 6))
    plt.plot(n_list, min_ratios, label='Min Ratio', marker='o')
    plt.plot(n_list, max_ratios, label='Max Ratio', marker='s')
    plt.plot(n_list, avg_ratios, label='Average Ratio', marker='^')
    
    plt.title('Execution Time Ratios by Input Size (n)')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Ratio (Time1 / Time2)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./ratio_plot2.jpg')
    plt.close()



if __name__ == "__main__":
    print("--- Problem 1 ---")
    problem_1a()
    problem_1b()