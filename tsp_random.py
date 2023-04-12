import random as rd
import chart
import map
import data
import time

MAX_ITER = 100000  # 최대 반복
POP_SIZE = 1  # 한 세대당 염색체 개수
distance = []  # 장소간 거리 2차원 테이블

class Chromosome:  # 염색체
    def __init__(self, size, g=None):
        # genes = 1차원 배열[0,1,2....998,999] = 방문 순서 -> 0~999 순서로 순회하고 0으로 돌아온다는 뜻
        self.genes = None
        self.fitness = 0 # 적합도 = 경로의 거리

        if g == None:
            self.genes = [*range(size)]
            rd.shuffle(self.genes)  # 무작위로 순서변경
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

    def calFitness(self):  # 적합도 계산
        self.fitness = 0
        for i in range(len(self)):  # 도시들간 거리 더하기
            self.fitness += distance[self.genes[i]][self.genes[i-1]]
        return self.fitness


class Population:  # 한 세대(염색체들을 가지고있음)
    def __init__(self, popSize, chroSize, pop=None):
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

class GeneticAlgorithm:
    def __init__(self, pop):
        self.population = pop
        self.fitnessMean = []  # 평균 적합도 추이
        self.fitnessBest = []  # 첫번째 염색체의 적합도 추이
        self.bestGene = pop[0]
        self.worstGene = pop[-1]

    def setPopulation(self, pop):
        self.population = pop

    def getProgress(self):  # 차트를 위한 추이 반영 - 알고리즘과는 상관없음
        fitnessSum = 0  # 평균을 구하기 위한 합계
        for c in self.population:
            fitnessSum += c.getFitness()
        self.fitnessMean.append(fitnessSum / POP_SIZE)  # 세대의 평균 적합도 추이
        # 세대의 적합도가 가장 좋은 염색체의 적합도 추이
        self.fitnessBest.append(self.population[0].getFitness())

        # 최단거리 최장거리 갱신
        if self.bestGene.getFitness() > self.population[0].getFitness():
            self.bestGene = self.population[0]
        if self.worstGene.getFitness() < self.population[-1].getFitness():
            self.worstGene = self.population[-1]

    def getBestGene(self):
        return self.bestGene

    def getWorstGene(self):
        return self.worstGene

    def drawResultChart(self, generation):  # 차트를 그리는 함수
        chart.drawChart(self.fitnessMean, self.fitnessBest, generation, self.bestGene, self.worstGene)


def main():  # 메인함수
    global distance
    distance = data.getDistanceList()  # 도시간 거리 2차원 테이블 가져오기
    cityNum = len(distance) # 도시 개수 (1000개)

    population = Population(POP_SIZE, cityNum)  # 무작위로 한 세대 생성
    cityMap = map.loadMap()  # 시각화 이미지 로드

    generation = 0
    ga = GeneticAlgorithm(population)  # 유전 알고리즘 인스턴스 생성
    start = time.time()  # 실행시간 측정
    while 1:
        generation += 1

        if generation == MAX_ITER:  # limit까지 도달할 경우
            ga.getProgress()  # 차트를 위해 평균 적합도, 최적 적합도 추이 반영
            break

        # 기존 집합을 새로운 염색체 집합으로 교체
        ga.setPopulation(Population(POP_SIZE, cityNum))

        ga.getProgress()  # 차트를 위해 평균 적합도, 최적 적합도 추이 반영

        # 10번마다 UI 업데이트
        if generation % 10 == 0 and not map.updateUI(cityMap, generation, ga.bestGene, time.time() - start):
            break

    t = time.time() - start
    print(f"실행시간 : {int(t//60)}분 {t%60}초")
    ga.drawResultChart(generation)  # 마지막으로 차트 그리기
    print("최적의 거리 :", ga.getBestGene().getFitness())
    # main함수 끝

if __name__ == '__main__':
    main()