import data
import random as rd

cities_x_y = data.getCities() # 장소간 거리 2차원 테이블
SortedAdjCities = data.getSortedAdjacentCities() # 각 도시에서 가까운 도시 리스트 -> [3][5] 3번 도시에서 5번째로 가까운 도시

def greedyDfs(start, size, pro): # 한 시작도시에서 가까운 도시들 위주로 size(2~50) 크기의 부분경로를 생성
    global SortedAdjCities
    route = [start]
    visited = [False] * len(SortedAdjCities)
    visited[start] = True

    now = start
    for _ in range(1, size): #  항상 가까운 도시만 방문하면 항상 같은 경로를 반환하기 때문에 국소최적해에 빠질 수 있으니
        for city in SortedAdjCities[now]: # 좀 더 먼곳도 탐색할 수 있도록 하되 가까운 곳에 가중치를 더 둠
            if not visited[city] and rd.random() < pro: # pro가 0.5면 가장 가까운 도시의 확률 = 0.5
                route.append(city)                              # 2번째로 가까운 도시 확률 = 0.25, 3번쨰는 = 0.125
                visited[city] = True
                now = city
                break
    return route # 가까운 도시들 위주로 구성된 부분경로 반환

def checkIntersection(line1, line2): # 두 선이 교차하는지 체크
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    den = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
    if den == 0:
        return False
    ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3))/den
    ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3))/den
    if ua > 0 and ua < 1 and ub > 0 and ub < 1:
        return True
    else:
        return False

def getIntersectCityIdx(route): # 다른 선과 교차하는 선을 가진 점들 찾아 반환
    added = [False] * len(route)
    idx = []
    for i in range(len(route)):
        for j in range(i+2, len(route)):
            if checkIntersection([cities_x_y[route[i]], cities_x_y[route[i-1]]], [cities_x_y[route[j]], cities_x_y[route[j-1]]]):
                for c in [i, i-1, j, j-1]: 
                    if not added[c] and 0 <= c < len(route) - 2: # 1000보다 2 작은 값 반환해야 cross2가 2이상 나올 수 있음
                        idx.append(c)
                        added[c] = True
            if len(idx) >= 15:
                return idx
    return idx