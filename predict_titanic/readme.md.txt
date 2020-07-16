###Sex
 - male은 0, female은 1로 대체

###Embarked
 - 결측값 : 가장 많이 나타나는 'S'로 채움
 - 'C', 'S', 'Q'를 각각 숫자로 변환(LabelEncoder())

###Fare
 - 결측값 : 평균값으로 채움
 - 5(0~4, 5~9, ...)씩 구간을 나눔

###Name
 - Name 추출
 - 표시는 다르지만 비슷한 것끼리 그룹화

###Age
 - 결측값 : Age 평균인 28로 채움
 - 구간을 나눠주기 위해 5(0~4, 5~9, ...)씩 나눔

###Feature selection
 - Ticket, Cabin은 사용하지 않음