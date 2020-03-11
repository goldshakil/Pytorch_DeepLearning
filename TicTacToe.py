# Problem 1: tick tack toe
import random
#################################################################################################################################
#check if the game is finished or not -> 0 (goingon)/ 1(draw)/2 (X/First Player)/3 (O/Second Player)
def check_end(arr):
    #column winner
    if (arr[0][0]==arr[1][0] and arr[1][0]==arr[2][0]):#column1
        if (arr[0][0]=='X'):
            return 2
        elif (arr[0][0]=='O'):
            return 3

    elif (arr[0][1]==arr[1][1] and arr[1][1]==arr[2][1]):#column2
        if (arr[0][1]=='X'):
            return 2
        elif (arr[0][1]=='O'):
            return 3

    elif (arr[0][2]==arr[1][2] and arr[1][2]==arr[2][2]):#column3
        if (arr[0][2]=='X'):
            return 2
        elif (arr[0][2]=='O'):
            return 3
    #Row Winner
    elif (arr[0][0]==arr[0][1] and arr[0][1]==arr[0][2]):#row1
        if (arr[0][0]=='X'):
            return 2
        elif (arr[0][0]=='O'):
            return 3

    elif (arr[1][0]==arr[1][1] and arr[1][1]==arr[1][2]):#row2
        if (arr[1][0]=='X'):
            return 2
        elif (arr[1][0]=='O'):
            return 3

    elif (arr[2][0]==arr[2][1] and arr[2][1]==arr[2][2]):#row3
        if (arr[2][0]=='X'):
            return 2
        elif (arr[2][0]=='O'):
            return 3
    #Diagonal Winner
    elif (arr[0][0]==arr[1][1] and arr[1][1]==arr[2][2]):#diagonal 1
        if (arr[0][0]=='X'):
            return 2
        elif (arr[0][0]=='O'):
            return 3

    elif (arr[0][2]==arr[1][1] and arr[1][1]==arr[2][0]):#diagonal2
        if (arr[0][2]=='X'):
            return 2
        elif (arr[0][2]=='O'):
            return 3

#On Going
    counter=0
    for i in range(0,3):
        for j in range(0,3):
            if(arr[i][j]=='X'or arr[i][j]=='O'):
                counter+=1
    if(counter!=9):
        return 0

#Draw
    return 1

##################################################################################################################################
#game handling function
def game(player):
    rows, cols = (3, 3)
    arr =[['-' for i in range(cols)] for j in range(rows)]

    switcher=player #used to switch between the player and pc

    player_char='-'
    PC_char='-'
    if (player==1): #player should start
        player_char='X'
        PC_char='O'
    else:
        player_char='O'
        PC_char='X'


    while (check_end(arr)==0):
        if (switcher==1):
            print("Player's Turn")
            #Player Turn
            #Take Input
            x=input("Enter X Position(0-2):")
            y=input("Enter Y Position(0-2):")
            arr[int(x)][int(y)]=player_char

            #Print Array
            for i in range(0,3):
                print(arr[i])
            switcher=0 #switch turn
        else:
            print("PC's Turn")
            #PC Turn
            #Take Input
            while (1):
                random_i=random.randint(0,2)
                random_j=random.randint(0,2)
                if (arr[random_i][random_j]=='-'):
                    arr[random_i][random_j]=PC_char
                    break;
                #Print Array
            for i in range(0,3):
                print(arr[i])
            switcher=1 #switch turn

    print("############### Game End ###############")

    if(check_end(arr)==1):
        print("DRAW!")
    elif(check_end(arr)==2):#first player/ X won
        if(player_char=='X'):
            print("YOU HAVE WON")
        else:
            print("PC HAS WON")
    elif(check_end(arr)==3):#second player/ O won
        if(player_char=='O'):
            print("YOU HAVE WON")
        else:
            print("PC HAS WON ")



#################################################################################################################################
#Driver Function
def main():
    #Choose the Player to start the game -> 1 player/ 0 PC
    chooser=random.randint(0,1)
    if(chooser==0):
        print("PC Will Start/ X Will Start")
    else:
        print("Player Will Start/ X Will Start")
    game(chooser)
##################################################################################################################################
#Driver Function Call
main()
