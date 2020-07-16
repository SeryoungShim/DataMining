import numpy as np
import sys
import csv

'''open file'''
#open csv file
filename = (sys.argv[1])
min_support = float(sys.argv[2])
min_confidence = float(sys.argv[3])
data = open(filename, mode='r', buffering=-1, encoding='utf-8', newline=None)
#data = np.genfromtxt(filename, encoding='ascii', dtype=np.int64, delimiter=",")

lines = []
C1 = [0 for i in range(100)]
i=0
for line in data:
    line = line.strip('\n')
    #lines.append(line.split(","))
    lines.append([int(i) for i in line.split(',')])

length = len(lines)
# Step 1: Find frequent 1-itemsets
for i in range(length):
    lines[i][1:].sort()
    for j in range(1, len(lines[i])):
        k = lines[i][j]
        C1[k] = C1[k] + 1               #count


#frequent 1-itemsets
F1 = [i for i in range(len(C1)) if (C1[i]/length)>=min_support]

k=0
C2_item = [[0]*2 for i in range((len(F1)-1)*len(F1)//2)]
# Step 2: Generate candidate 2-itemsets
for i in range(len(F1)):
    for j in range(i+1, len(F1)):
        C2_item[k][0] = i
        C2_item[k][1] = j
        k += 1

# Step 3: Find frequent 2-itemsets
F2_item = np.full((100,100), 0)
F2 = []
for i in range(length):
    for j in range(len(C2_item)):
        if C2_item[j][0] in lines[i]:
            if C2_item[j][1] in lines[i]:
                F2_item[(C2_item[j][0])][(C2_item[j][1])] += 1

for i in range(len(F2_item)):
    for j in range(len(F2_item[i])):
        if(F2_item[i][j]/length>=min_support):
            F2.append([i, j])


print("Association rules found:")
# Step 4: Generate association rules
association_rule = []
for i in range(len(F2)):
    if (F2_item[F2[i][0]][F2[i][1]]/C1[F2[i][0]] >= min_confidence):
        association_rule.append([F2[i][0], F2[i][1]])
        print(F2[i][0], " -> ", F2[i][1], " (support = ", F2_item[F2[i][0]][F2[i][1]]/length, ", confidence = ", F2_item[F2[i][0]][F2[i][1]]/C1[F2[i][0]],end='')
        print(")")
    if (F2_item[F2[i][0]][F2[i][1]]/C1[F2[i][1]] >= min_confidence):
        association_rule.append([F2[i][1], F2[i][0]])
        print(F2[i][1], " -> ", F2[i][0], " (support = ", F2_item[F2[i][0]][F2[i][1]]/length, ", confidence = ", F2_item[F2[i][0]][F2[i][1]]/C1[F2[i][1]],end='')
        print(")")
             


