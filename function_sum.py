def ksum (stop,start=1):
    _sum =0
    for i  in range(start,stop+1):
        _sum+=i
    return _sum

def ksum (stop,start=1,step=1):
    _sum =0
    for i  in range(start,stop+1,step):
        _sum+=i
    return _sum
'''
res = ksum(start=100,stop=1000)
print (res)
res = ksum(start=100,stop=1000,step=2)
print (res)
'''
