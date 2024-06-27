def fibonacci (stop):
    fibonacci1 = 1
    fibonacci2 = 1
    void = 0
    print(fibonacci1,end=' ')
    for i  in range(stop-1):
        print(fibonacci2,end=' ')
        void = fibonacci2
        fibonacci2=fibonacci1+fibonacci2
        fibonacci1=void
    return fibonacci
fibonacci(6)
