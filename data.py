import numpy as np
import csv
import copy

adjCities = None
cities = []
dist = None

def distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist

def getDistanceList(): # 도시간 거리 2차원 테이블 반환
    global dist, cities
    if dist != None:
        return copy.deepcopy(dist)
    
    dist = [[None] * len(cities) for _ in range(len(cities))]
    for i in range(len(cities)):
        for j in range(len(cities)):
            dist[i][j] = distance([cities[i][0], cities[i][1]], [cities[j][0], cities[j][1]])
    return copy.deepcopy(dist)

def getCities(): # 도시들의 좌표 반환
    return copy.deepcopy(cities)

def getSortedAdjacentCities(): # 정렬된 인접도시 2차원 리스트 반환
    global dist, adjCities     # adjCities[3][5] = 3번 도시에서 5번째로 가까운 도시
    if adjCities != None:       # adjCities[1][0] = 1번 도시에서 0번째로 가까운 도시 = 자기자신인 1번
        return copy.deepcopy(adjCities)

    if dist == None:
        getDistanceList()

    adjCities = copy.deepcopy(dist) # 각 거리에 도시 인덱스 정보도 추가해 정렬하여
    for i in range(len(adjCities)): # 가까운 순으로 정렬된 도시 번호 얻기
        for j in range(len(adjCities)):
            adjCities[i][j] = [adjCities[i][j], j] #
        adjCities[i].sort()

    for i in range(len(adjCities)):
        for j in range(len(adjCities)):
            adjCities[i][j] = adjCities[i][j][1]
    return copy.deepcopy(adjCities)

with open('2023_AI_TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        cities.append(list(map(float, row)))

def test(sol):
    total_cost = 0
    for idx in range(len(sol)):
        p1 = [cities[sol[idx]][0], cities[sol[idx]][1]]
        p2 = [cities[sol[idx-1]][0], cities[sol[idx-1]][1]]

        dist = distance(p1, p2)

        total_cost += dist
    return total_cost

print("data loading..")