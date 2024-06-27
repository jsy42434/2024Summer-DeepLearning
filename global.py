def cntt():
    global count
    count = 100
    print('count =', count)

cntt()
count = 300
print('count =', count)
cntt()
print('count =', count)

