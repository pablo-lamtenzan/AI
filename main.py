

class Obj :
    def __init__ (self, a) :
        self.variable = a

    def method(self) :
        for i in range (10) :
            self.variable+=1


class Jbo(Obj) :
    def __init__(self) :
        Obj.__init__(self, 30)

def __main__() :
    e = Jbo()
    e.method()
    print (e.variable)
    return


__main__()