import csv
f = open('score.csv','w',encoding='cp949',newline='')
wr = csv.writer(f)

sn=5
student=[]
for i in range(1,sn+1):
    score=[]
    score.append(input(str(i)+'번째 학생 이름:'))
    score.append(input(score[0]+'의 국어성적'))
    score.append(input(score[0]+'의 수학성적'))
    score.append(input(score[0]+'의 영어성적'))
    print('-'*70)
    student.append(score)


print('순번\t이름\t국어\t수학\t영어\t합계\t평균\t성적')
wr.writerow(['순번','이름','국어','수학','영어','합계','평균','성적'])
print('-'*70)

n=0
allavg=0
for i in student:
    n+=1
    score2=[]
    score2.extend([int(i[1]),int(i[2]),int(i[3])])
    ssum=sum(score2)
    savg=ssum/3
    allavg+=savg
    grade=''
    if savg >=90 and savg <=100:
        grade='A'
    elif savg >=80:
        grade='B'
    elif savg >=70:
        grade='C'
    elif savg >=60:
        grade='D'
    else:
        grade='F'
    print(n,'\t',i[0],'\t',i[1],'\t',i[2],'\t',i[3],'\t',ssum,'\t',round(savg,2),'\t',grade)
    wr.writerow([n,i[0],i[1],i[2],i[3],ssum,round(savg,2),grade])


    
allavg=allavg/sn
print('-'*70)
print('전체 평균\t',round(allavg,2))
wr.writerow(['전체 평균',round(allavg,2)])


f.close()
