import copy
import datetime
import math
from math import sin
import time
from numpy import matrix
from numpy import linalg

# Лабораторная 1, Задание 6.1
# вычисление времени счета
import numpy as np

start_time = datetime.datetime.now()

a = 1
b = 9
eps = 0.001


def f(x):
    return pow(2, -x) - sin(x)


def otdelenie(k, j):
    print('Определение интервалов с корнями уравнения:')
    for i in range(k, j):
        if f(i) * f(i + 1) <= 0:
            print('найденный интервал с корнем:')
            print(i, i + 1)
            new_a = i
            new_b = i + 1
            print('точность определения корня:')
            precision: float = 1 / 2 * (i + 1 - i)
            print(precision)


def dyhotomy(a, b, eps):
    print('Решение уравнения методом дихотомии:')
    root = None
    n = 1
    while abs(f(b) - f(a)) > eps:
        mid = (a + b) / 2
        n += 1
        if f(mid) == 0 or abs(f(mid)) < eps:
            root = mid
            break
        elif f(a) * f(mid) < 0:
            b = mid
            print(f'Промежуточное значение функции fx{n}')
            print(f'{f(mid)}')
        else:
            a = mid
            print(f'Промежуточное значение функции fx{n}')
            print(f'{f(mid)}')
    if root is not None:
        # вычисление разницы времени начала отсчета и времени окончания вычислений
        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000
        pogreshn_korn = (b - a) / 2
        print(f'Корень уравнения в точке x = {root}')
        print(f'Значение функции {f(root)}')
        print(f'Количество итераций {n}')
        print(f'Погрешность корня =  {pogreshn_korn}')
        print("--- %s секунд ---" % execution_time)


otdelenie(a, b)
dyhotomy(6, 7, eps)

# Лабораторная 1, Задание 6.2 Решение СЛАУ методом Гаусса
# Вариант 3
#   {4. 3, 2, 1}  основная матрица
#   {3, 6, 4, 2}
#   {2, 4, 6, 3}
#   {1, 2, 3, 4}
#   {3}             дополнение
#   {6}
#   {4}
#   {7}

A = [
    [4, 3, 2, 1],
    [3, 6, 4, 2],
    [2, 4, 6, 3],
    [1, 2, 3, 4]
]

B = [3, 6, 4, 7]


def gauss_(aa, bb):
    n = len(aa)
    sgn = 1
    for r in range(n):  # r - номер опорной строки
        z = aa[r][r]  # опорный элемент
        # перебор всех строк, расположенных ниже r
        if abs(z) < 1.0e-10:  # ноль на диагонали
            # ищем ненулевой элемент ниже
            for j in range(r + 1, n):
                if abs(aa[j][r]) > 1.0e-10:
                    for jj in range(r, n):
                        aa[j][jj], aa[r][jj] = aa[r][jj], aa[j][jj]
                    bb[j], bb[r] = bb[r], bb[j]
                    z = aa[r][r]
                    sgn = -sgn
                    break
            else:
                return None
        for i in range(r + 1, n):
            q = aa[i][r] / z
            for j in range(n):
                aa[i][j] = aa[i][j] - aa[r][j] * q
            bb[i] = bb[i] - bb[r] * q
    return (aa, bb, sgn)


# Вычисление главного определителя

def det_tri(a, sgn=1):
    n = len(a)
    p = sgn
    for i in range(n):
        p = p * a[i][i]
    return p


# Метод Гаусса (обратный ход)
iteration = 0


def rev_calc(a, b):
    global iteration
    n = len(b)
    res = [0 for _ in range(n)]
    i = n - 1
    res[i] = b[i] / a[i][i]
    i = i - 1
    while (i >= 0):
        s = b[i]
        for j in range(i + 1, n):
            s = s - a[i][j] * res[j]
        res[i] = s / a[i][i]
        i = i - 1
        iteration = iteration + 1
    return res, iteration


# Запуск тестов


res = gauss_(A, B)

if res is None:
    print("Нет решений!")
else:
    print(res[0])
    print(res[1])
    print(det_tri(res[0], res[2]))
    print(rev_calc(res[0], res[1]))
    print('Количество итераций', iteration)

# 6.2.1 Решение СЛАУ: Метод Гаусса (обновленный)
m = len(A)
x = [0. for i in range(m)]
Iteration = 0
converge = False
eps = 0.
while not converge:
    x_new = np.copy(x)
    for i in range(m):
        s1 = sum(A[i][j] * x_new[j] for j in range(i))
        s2 = sum(A[i][j] * x[j] for j in range(i + 1, m))
        x_new[i] = (B[i] - s1 - s2) / A[i][i]
    eps = sum(abs(x_new[i] - x[i]) for i in range(m))
    converge = eps < 1e-6
    Iteration += 1
    x = x_new
print('Количество итераций :', Iteration)
print('Решение системы уравнений :', x)
print('Погрешность :', eps)

print("Проверка решения через numpy:", np.linalg.solve(A, B))  # проверка решения с помощью доп. библиотеки numpy

# 6.2.2 Вычисление определителя матрицы методом миноров (декомпозиции)

A = [
    [4, 3, 2, 1],
    [3, 6, 4, 2],
    [2, 4, 6, 3],
    [1, 2, 3, 4]
]


def det2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def minor(matrix, i, j):
    tmp = [row for k, row in enumerate(matrix) if k != i]
    tmp = [col for k, col in enumerate(zip(*tmp)) if k != j]
    return tmp


def determinant(matrix):
    size = len(matrix)
    if size == 2:
        return det2(matrix)

    return sum((-1) ** j * matrix[0][j] * determinant(minor(matrix, 0, j))
               for j in range(size))


print('Определитель матрицы А: ', determinant(A))

# 6.2.3

A = [
    [4, 3, 2, 1],
    [3, 6, 4, 2],
    [2, 4, 6, 3],
    [1, 2, 3, 4]
]

E = [[1, 0, 0, 0],  # Единичная матрица для обратного хода
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]

WholeMatrix = [[0 for i in range(8)]] * 4
print(WholeMatrix)


def invert_matrix(AM, IM):
    for fd in range(len(AM)):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(len(AM)):
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        print(IM[fd])
        for i in list(range(len(AM)))[0:fd] + list(range(len(AM)))[fd + 1:]:
            crScaler = AM[i][fd]
            for j in range(len(AM)):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
            print(IM[i][j])
    return IM


print('Обратная матрица:')
print(invert_matrix(A, E))

print(np.linalg.inv(E))
