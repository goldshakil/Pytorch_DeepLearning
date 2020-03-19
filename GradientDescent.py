# Sample Gradient Descent Method
# Change times of looping and learning rate for better results
def main():
    learning_rate=0.01
    w0=0.9
    w1=0.9
    w2=0.9

    looper=10000

    print("Number of Looping:%d " % looper)
    a=0.0
    b=0.0
    c=0.0

    for i in range(0,looper):
        a=w0-learning_rate*(8*w0+8*w1+12*w2-8)
        b=w1-learning_rate*(8*w0+12*w1+20*w2-10)
        c=w2-learning_rate*(12*w0+20*w1+36*w2-14)
        w0=a
        w1=b
        w2=c


    print("Weight0:%f" % w0)
    print("Weight1:%f" % w1)
    print("Weight2:%f" % w2)



main()
