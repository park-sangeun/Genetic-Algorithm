import cv2
import data
import copy

cities = []
IMG_HEIGHT = 600
IMG_WIDTH = 600

def loadMap():
    cityMap = cv2.imread('white.png')
    cityMap = cv2.resize(cityMap, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA,)
    global cities
    cities = data.getCities()
    for i in range(len(cities)):
        cities[i][0] = int(50 + cities[i][0] * 5)
        cities[i][1] = int((100 - cities[i][1]) * 5 + 50)

    # 도시들이 있는 공간을 구분하는 사각형 만들기
    cv2.line(cityMap, pt1=(50, 550), pt2=(50, 50), color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(cityMap, pt1=(50, 550), pt2=(550, 550), color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(cityMap, pt1=(50, 50), pt2=(550, 50), color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(cityMap, pt1=(550, 550), pt2=(550, 50), color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)

    cv2.putText(cityMap, org=(100, 30), text='Visualization of TSP Cities', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(cityMap, org=(450, 575), text='Press Q to exit', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # 도시들 파란 점으로 표시, 시작점은 빨간색
    cv2.circle(cityMap, center=(cities[0][0], cities[0][1]), radius=5, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA,)
    for x, y in cities[1:]:
        cv2.circle(cityMap, center=(x, y), radius=2, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

    cv2.imshow('cityMap', cityMap)
    cv2.waitKey(100)
    return cityMap

def updateUI(prevMap, generation, bestGene): # UI 갱신
    global cities
    cityMap = prevMap.copy()

    for i in range(len(bestGene)): # 장소들을 잇는 선 그리기
        cv2.line(cityMap,
            pt1=(cities[bestGene[i-1]][0], cities[bestGene[i-1]][1]),
            pt2=(cities[bestGene[i]][0], cities[bestGene[i]][1]),
            color=(100, 100, 100),
            thickness=1,
            lineType=cv2.LINE_AA
        )
    # 아래 텍스트 갱신
    cv2.putText(cityMap, org=(50, 575), text='Generation: %d' % generation, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(cityMap, org=(200, 575), text='Distance: %.2f' % bestGene.getFitness(), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('cityMap', cityMap)

    if cv2.waitKey(1) == ord('q'): # 대기시간
        return False
    return True