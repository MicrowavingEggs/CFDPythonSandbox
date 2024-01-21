from numba import jit

def parmparse():
    res = dict()
    f = open("./inputs.txt")
    for line in f:
        data = line.split(":")
        key = data[0]
        value = data[1]
        if "FLAG" in key: #Then it's a bool
            value = bool(int(value))
        elif len(value.split(".")) > 1 : #Then it's a float
            value = float(value)
        else: #Otherwise it's an INT
            value = int(value)
        res[key] = value
    return res

@jit(nopython=True)
def Sum(a,b): #a: array; b: array
    return a+b

@jit(nopython=True)
def Dot(a,b): #a: array; b: array
    return a*b


print(parmparse())