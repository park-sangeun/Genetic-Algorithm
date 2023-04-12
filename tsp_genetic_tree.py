import random as rd
import chart
import map
import data
import time
import additional_algorithm

MAX_ITER = 1000000 # 최대 반복
POP_SIZE = 30 # 한 세대당 염색체 개수
MUT_RATE = 0.13 # 돌연변이 확률
distance = [] # 장소간 거리 2차원 테이블

class Chromosome: # 염색체 
    def __init__(self, size, g=None):
# genes = 1차원 배열[0,1,2....998,999] = 방문 순서 -> 0~999 순서로 순회하고 0으로 돌아온다는 뜻
        self.genes = None
        self.fitness = 0  # 적합도 = 경로의 거리

        if g == None:
            self.genes = [*range(size)]
            rd.shuffle(self.genes) # 무작위로 순서변경
        else:
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
        for i in range(len(self)): # 도시들간 거리 더하기
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

    def sortPop(self):
        self.pop.sort(key = lambda x: x.getFitness())

class GeneticAlgorithm:
    def __init__(self, pop):
        self.population = pop # Population
        self.fitnessMean = [] # 평균 적합도 추이
        self.fitnessBest = [] # 첫번째 염색체의 적합도 추이
        self.bestGene = pop[0] # 적합도가 제일 좋은 염색체
        self.worstGene = pop[-1] # 적합도가 제일 나쁜 염색체
        self.interSectCity = [] # 경로가 교차하는 선을 가지고 있는 점(도시)들

    def setPopulation(self, pop):
        self.population = pop

    def getIntersectCities(self):
        self.interSectCity = additional_algorithm.getIntersectCityIdx(self.bestGene.getGene())

    def sortPopulation(self):
        self.population.sortPop()
    
    def select(self): # 부모 선택
        # 적합도에 비례해 선택하면 안좋은 개체를 선택할 확률이 너무 높으므로 좋은 염색체를 남길 확률 더 증가
        # 2 ** i 면 제일 좋은 것이 확률 1/2 -> [1/2, 1/4, 1/8, 1/16...]
        return rd.choices(self.population, weights= [1.4 ** i for i in range(POP_SIZE - 1, -1, -1)])[0]
    
    def crossover(self, returnTwo = True): # 교배
        # https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cni1577&logNo=221237605486 참고
        father = self.select()
        mother = self.select()

        cross1 = rd.randrange(len(father) - 2)
        if rd.random() < 0.5 and len(self.interSectCity) > 0: # 확률적으로 경로가 교차하는 선을 가지고 있는 점(도시) 선택
            cross1 = rd.choice(self.interSectCity)

        cross2 = rd.randrange(cross1 + 2, min(cross1 + 50, len(father))) # cross1 보다 2 ~ 50 더 큰 수
        fMid = additional_algorithm.greedyDfs(father[cross1], cross2 - cross1, 0.5) # 휴리스틱함수로 크기가 2~50인 교환 구역 생성
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

    def mutate(self): # 돌연변이
        for chro in self.population:
            if rd.random() < MUT_RATE:
                a, b = sorted(rd.sample(range(len(chro)), 2))
                if rd.random() < 0.5 and len(self.interSectCity) > 0: # 특정 확률로 교차 선들을 가진 점끼리 변경
                    a, b = sorted(rd.sample(self.interSectCity, 2))
                if rd.random() < 0.5: # 상호 교환 연산자 - 단순히 두 도시 변경
                    chro.getGene()[a], chro.getGene()[b] = chro.getGene()[b], chro.getGene()[a]
                    chro.calFitness()
                else: # 역치 연산자 - 두 점을 선택 후 그 사이의 순서 변경
                    if a != 0:
                        chro.setGene(chro[:a] + chro[b-1:a-1:-1] + chro[b:])
                    else:
                        chro.setGene(chro[b-1::-1] + chro[b:])

    def getProgress(self): # 차트를 위한 추이 반영 - 알고리즘과는 상관없음
        fitnessSum = 0 # 평균을 구하기 위한 합계
        for c in self.population:
            fitnessSum += c.getFitness()
        self.fitnessMean.append(fitnessSum / POP_SIZE) # 세대의 평균 적합도 추이
        self.fitnessBest.append(self.population[0].getFitness()) # 세대의 적합도가 가장 좋은 염색체의 적합도 추이

        if self.bestGene.getFitness() > self.population[0].getFitness(): # 최단거리 최장거리 갱신
            self.bestGene = self.population[0]
        if self.worstGene.getFitness() < self.population[-1].getFitness():
            self.worstGene = self.population[-1]

    def getBestGene(self):
        return self.bestGene
    
    def getWorstGene(self):
        return self.worstGene

    def drawResultChart(self, generation): # 차트를 그리는 함수
        chart.drawChart(self.fitnessMean, self.fitnessBest, generation, self.bestGene, self.worstGene)

    def saveSolution(self): # 파일에 최적 경로를 저장하는 함수
        f = open("solution_05.csv", "w")
        startCityIdx = self.bestGene.getGene().index(0) # 0번 도시의 인덱스
        route = self.bestGene.getGene()[startCityIdx:] + self.bestGene.getGene()[:startCityIdx]
        for city in route:
            f.write(str(city)+'\n')
        f.close()
        print("최적 경로가 solution_05.csv 파일에 저장되었습니다.")


def main(): # 메인함수
    global distance, SortedAdjCities
    distance = data.getDistanceList() # 도시간 거리 2차원 테이블 가져오기
    SortedAdjCities = data.getSortedAdjacentCities() # 인접 도시 리스트 가져오기
    cityNum = len(distance) # 도시 개수 (1000개)

    population = Population(POP_SIZE, cityNum) # 무작위로 한 세대 생성

    cityMap = map.loadMap() # 시각화 이미지 로드

    generation = 0
    ga = GeneticAlgorithm(population) # 유전 알고리즘 인스턴스 생성
    ga.sortPopulation() # 적합도 순으로 정렬
    start = time.time() # 실행시간 측정
    while 1:
        generation += 1
        
        if generation == MAX_ITER: # limit까지 도달할 경우
            ga.getProgress() # 차트를 위해 평균 적합도, 최적 적합도 추이 반영
            break

        if generation % 100 == 1: # 100번마다 교차하는 선들을 가진 점을 탐색
            ga.getIntersectCities()
    
        new_pop = []
        for _ in range(POP_SIZE // 2): # n/2번 2자식씩 생성하며 교배를 하여 새로운 염색체 생성
            child1, child2 = ga.crossover()
            new_pop.append(Chromosome(cityNum, child1))
            new_pop.append(Chromosome(cityNum, child2))

        ga.setPopulation(Population(POP_SIZE, cityNum, new_pop)) # 기존 집합을 새로운 염색체 집합으로 교체

        ga.mutate() # 돌연변이 연산 수행
        ga.sortPopulation()
        ga.getProgress() # 차트를 위해 평균 적합도, 최적 적합도 추이 반영

        # 10번마다 UI 업데이트
        if generation % 10 == 0 and not map.updateUI(cityMap, generation, ga.bestGene, time.time() - start):
            break

    t = time.time() - start
    print(f"실행시간 : {int(t//60)}분 {t%60}초")
    ga.drawResultChart(generation) # 마지막으로 차트 그리기
    print("최적의 거리 :", ga.getBestGene().getFitness())
    # ga.saveSolution() # 파일에 최단거리 저장
    # main함수 끝

if __name__ == '__main__':
    main()