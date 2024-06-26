import csv

f = open('score.csv')
data = csv.reader(f)
k = next(data)
print('순번\t이름\t국어\t수학\t영어\t합계\t평균\t성적')
print('-'*70)
for line in data:
    for i in line:
        print(i,end='\t')
    print('')
f.close()
