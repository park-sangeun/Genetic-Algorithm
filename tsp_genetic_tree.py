import random as rd
import chart
import map
import data
import time

MAX_ITER = 100000 # 최대 반복
POP_SIZE = 30 # 한 세대당 염색체 개수
MUT_RATE = 0.13 # 돌연변이 확률
NEAR_WEIGHT = 0.5 # 휴리스틱 함수의 가중치 (ctr버튼을 누른상태에서 NEAR_WEIGHT를 클릭하면 자세한 설명이 나옵니다)
distance = [] # 장소간 거리 2차원 테이블
SortedAdjCities = [] # 서로 가까운 순으로 정렬된 장소간 거리 -> [1][5] = 1번 도시에서 5번째로 가까운 도시, [i][0] = 0

class Chromosome: # 염색체 
    def __init__(self, size, g=None):
# genes = 1차원 배열[0,1,2....998,999] = 방문 순서 -> 0~999 순서로 순회하고 0으로 돌아온다는 뜻
        self.genes = None
        self.fitness = 0

        if g == None:
            self.genes = [*range(size)]
            rd.shuffle(self.genes) # 무작위로 순서변경
        else:
            if(len(g) != 1000):
                print("누락됨")
            self.genes = g.copy()

        self.calFitness()

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, index):
        return self.genes[index]

    def __setitem__(self, key, value):
        self.genes[key] = value
        self.calFitness()

    def __repr__(self):
        return str(self.genes)

    def setGene(self, g):
        self.genes = g.copy()
        self.calFitness()

    def getGene(self):
        return self.genes

    def getFitness(self):
        return self.fitness

    def calFitness(self): # 적합도 계산
        self.fitness = 0
        for i in range(len(self)): # 중간 도시들 적합도 더하기
            self.fitness += distance[self.genes[i]][self.genes[i-1]]
        return self.fitness

class Population: # 한 세대(염색체들을 가지고있음)
    def __init__(self, popSize, chroSize, pop = None):
        self.pop = []

        if pop == None:
            for _ in range(popSize):
                self.pop.append(Chromosome(chroSize))
        else:
            self.pop = pop
    
    def __len__(self):
        return len(self.pop)

    def __getitem__(self, index):
        return self.pop[index]

    def __setitem__(self, key, value):
        self.pop[key] = value

    def __repr__(self): # 출력
        ret = ""
        print(len(self.pop))
        for i, chro in enumerate(self.pop):
            ret += f"염색체 # {i} = {chro} 적합도 = {chro.getFitness()}\n"
        return ret + '\n'

    def sortPop(self):
        self.pop.sort(key = lambda x: x.getFitness())

class GeneticAlgorithm:

    def __init__(self, pop):
        self.fitnessMean = [] # 평균 적합도 추이
        self.fitnessBest = [] # 첫번째 염색체의 적합도 추이
        self.bestGene = pop[0]
        self.worstGene = pop[-1]
    
    def select(self, pop): # 부모 선택
        # 적합도에 비례해 선택하면 안좋은 개체를 선택할 확률이 너무 높으므로 좋은 염색체를 남길 확률 더 증가
        # 2 ** i 면 제일 좋은 것이 확률 1/2 -> [1/2, 1/4, 1/8, 1/16...]
        return rd.choices(pop, weights= [1.4 ** i for i in range(POP_SIZE - 1, -1, -1)])[0]
    
    def huristic(self, start, size): # 한 시작도시에서 가까운 도시들 위주로 size(2~50) 크기의 부분경로를 생성
        global SortedAdjCities
        route = []
        visited = [False] * len(SortedAdjCities)
        visited[start] = True

        now = start
        for _ in range(size): #  항상 가까운 도시만 방문하면 항상 같은 경로를 반환하기 때문에 국소최적해에 빠질 수 있으니
            for city in SortedAdjCities[now]: # 좀 더 먼곳도 탐색할 수 있도록 하되 가까운 곳에 가중치를 더 둠
                if not visited[city] and rd.random() < NEAR_WEIGHT: # NEAR_WEIGHT가 0.5면 가장 가까운 도시의 확률 = 0.5
                    route.append(city)                              # 2번째로 가까운 도시 확률 = 0.25, 3번쨰는 = 0.125
                    visited[city] = True
                    now = city
                    break
        return route # 가까운 도시들 위주로 구성된 부분경로 반환
    
    def crossover(self, pop, returnTwo = True): # 교배
        # https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cni1577&logNo=221237605486 참고
        father = self.select(pop)
        mother = self.select(pop)

        cross1 = rd.randrange(len(father) - 2)
        cross2 = rd.randrange(cross1 + 2, min(cross1 + 50, len(father)))
        fMid = self.huristic(father[cross1], cross2 - cross1) # 휴리스틱함수로 교환 구역 생성
        mMid = mother[cross1:cross2]

        visitf = [False] * len(father) # 도시 중복 방문 방지용 체크배열
        visitm = [False] * len(mother)
        for i in fMid:
            visitf[i] = True
        for i in mMid:
            visitm[i] = True

        child1 = []
        child2 = []
        for i in range(len(father)):
            if len(child1) == cross1: # 교환 구역 시작점에 오면 교환 구역 추가
                child1 += mMid
            if len(child2) == cross1:
                child2 += fMid

            if not visitm[father[i]]: # 교환 구역에 없는 원소 추가
                child1.append(father[i])
            if not visitf[mother[i]]:
                child2.append(mother[i])

            if len(child1) == cross1: # 한번 더 체크해줘야 오류x
                child1 += mMid
            if len(child2) == cross1:
                child2 += fMid

        if returnTwo:
            return (child1, child2)
        else:
            return child1

    def mutate(self, pop): # 돌연변이
        for chro in pop:
            if rd.random() < MUT_RATE:
                if rd.random() < 0.5: # 상호 교환 연산자 - 단순히 두 도시 변경
                    a, b = rd.sample(range(len(chro)), 2)
                    chro.getGene()[a], chro.getGene()[b] = chro.getGene()[b], chro.getGene()[a]
                else: # 역치 연산자 - 두 점을 선택 후 그 사이의 순서 변경
                    a, b = sorted(rd.sample(range(len(chro) + 1), 2))
                    if a != 0:
                        chro.setGene(chro[:a] + chro[b-1:a-1:-1] + chro[b:])
                    else:
                        chro.setGene(chro[b-1::-1] + chro[b:])

    def getProgress(self, pop): # 차트를 위한 추이 반영 - 알고리즘과는 상관없음
        fitnessSum = 0 # 평균을 구하기 위한 합계
        for c in pop:
            fitnessSum += c.getFitness()
        self.fitnessMean.append(fitnessSum / POP_SIZE) # 세대의 평균 적합도 추이
        self.fitnessBest.append(pop[0].getFitness()) # 세대의 적합도가 가장 좋은 염색체의 적합도 추이

        if self.bestGene.getFitness() > pop[0].getFitness(): # 최단거리 최장거리 갱신
            self.bestGene = pop[0]
        if self.worstGene.getFitness() < pop[-1].getFitness():
            self.worstGene = pop[-1]

    def getBestGene(self):
        return self.bestGene
    
    def getWorstGene(self):
        return self.worstGene

    def drawResultChart(self, generation): # 차트를 그리는 함수
        chart.drawChart(self.fitnessMean, self.fitnessBest, generation, self.bestGene, self.worstGene)
        print("최적의 거리 :", self.bestGene.getFitness())

    def saveSolution(self): # 파일에 최적 경로를 저장하는 함수
        f = open("route.csv", "w")
        startCityIdx = self.bestGene.getGene().index(0) # 0번 도시의 인덱스
        route = self.bestGene.getGene()[startCityIdx:] + self.bestGene.getGene()[:startCityIdx]
        for city in route:
            f.write(str(city)+'\n')
        f.close()
        print("최적 경로가 route.csv 파일에 저장되었습니다.")

def main(): # 메인함수
    global distance, SortedAdjCities
    distance = data.getDistanceList() # 도시간 거리 2차원 테이블 가져오기
    SortedAdjCities = data.getSortedAdjacentCities()
    cityNum = len(distance)
    start = time.time() # 실행시간 측정

    population = Population(POP_SIZE, cityNum) # 무작위로 한 세대 생성
    population.sortPop()
    cityMap = map.loadMap() # 시각화 이미지 로드

    generation = 0
    ga = GeneticAlgorithm(population) # 유전 알고리즘 인스턴스 생성
    while 1:
        generation += 1
        
        if generation == MAX_ITER: # limit까지 도달할 경우
            ga.getProgress(population) # 차트를 위해 평균 적합도, 최적 적합도 추이 반영
            break
        
        new_pop = []
        for _ in range(POP_SIZE // 2): # n/2번 2자식씩 생성하며 교배를 하여 새로운 염색체 생성
            child1, child2 = ga.crossover(population)
            new_pop.append(Chromosome(cityNum, child1))
            new_pop.append(Chromosome(cityNum, child2))

        population = Population(POP_SIZE, cityNum, new_pop) # 기존 집합을 새로운 염색체 집합으로 교체

        ga.mutate(population) # 돌연변이 연산 수행
        population.sortPop()

        if generation % 10 == 0 and not map.updateUI(cityMap, generation, ga.bestGene): # 10번마다 UI 업데이트
            break
        
        ga.getProgress(population) # 차트를 위해 평균 적합도, 최적 적합도 추이 반영

    t = time.time() - start
    print(f"실행시간 : {int(t//60)}분 {t%60}초")
    ga.drawResultChart(generation) # 마지막으로 차트 그리기
    ga.saveSolution()
    # main함수 끝

if __name__ == '__main__':
    main()