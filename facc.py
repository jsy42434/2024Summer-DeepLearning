def facc(n):
    if n<=1:
        return 1
    else :
        return n * facc(n-1)

print('4! = ',facc(4))
