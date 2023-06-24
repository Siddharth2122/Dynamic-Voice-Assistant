import turtle

flag=turtle.Turtle()
for i in range(24):
    flag.setposition(0,0)
    flag.color("blue")
    flag.forward(149)
    flag.setposition(0,0)
    flag.right(15)

flag.setposition(0,-150)
flag.circle(151)
flag.penup()



#orange strip

flag.setposition(-500,150)
flag.pencolor("orange")
flag.pendown()
flag.fillcolor("orange")
flag.begin_fill()
for i in range(4):
    flag.forward(1000)
    flag.left(90)
flag.end_fill()
flag.penup()

#green strip

flag.setposition(-500,-150)
flag.pencolor("green")
flag.pendown()
flag.fillcolor("green")
flag.begin_fill()
for i in range(4):
    flag.forward(1000)
    flag.right(90)
flag.end_fill()
flag.penup()


flag.left(90)
flag.pendown()
flag.pencolor("blue")
flag.forward(300)
flag.penup()
flag.right(90)
flag.forward(1000)
flag.right(90)
flag.pendown()
flag.forward(300)
flag.penup()