#Connect4Learn

import numpy as np
import random
import os 




#Board logic ----- createBoard, displayBoard, clearBoard
def createBoard():
    return np.zeros((6,7),dtype=int)

board = createBoard()

def displayBoard():
    print(board)

def clearBoard():
    for x in range(10):
        print(" ")

#ErrorChecks
def typeCheck():
    while True:
        try:
            playerMove = int(input("Which column would you like to drop in?"))
            return playerMove
        except ValueError:
            print("Invalid character entered")

#Player move logic ----- getPlayerMove, getLowestRow,
def getPlayerMove():
    playerMove = typeCheck()
    while playerMove > 7 or playerMove < 1:
        print("That is not a column. Re enter")
        playerMove = typeCheck()
    return playerMove

def getLowestRow(playerMove):
    for row in range(5,-1,-1):
        if board[row][playerMove-1] == 0:
            return row
    return None
#AI move logic ----- getAIMove, (getLowestRow reapplies here), initMath, return calculation, learnFromGame flattenBoard,
memoryBoards = []
memoryMoves = []


explorationRate = 0.1

def getAIMove():
    global inputLayer, weights, bias, hiddenLayer, outputWeights, outputBias, outputLayer, memoryBoards, memoryMoves

    inputLayer = flattenBoard()
    for x in range(len(hiddenLayer)):
        calculation = 0
        for y in range(len(inputLayer)):
            result = inputLayer[y] * weights[x][y]
            calculation += result
        calculation += bias[x]
        hiddenLayer[x] = calculation


    for x in range(7):
        calculation = 0
        for y in range(len(hiddenLayer)):
            result = hiddenLayer[y] * outputWeights[x][y]
            calculation += result
        calculation += outputBias[x]
        outputLayer[x] = calculation

#return calculation
    if random.random() < explorationRate:
        columnChoice = random.randint(0,6)
    else:
        columnChoice = outputLayer.index(max(outputLayer))
    memoryBoards.append(inputLayer.copy())
    memoryMoves.append(columnChoice)
    return columnChoice

#----- IF MATH DOES NOT EXIST (FILE)
def initMath():
    global weights, bias, outputWeights, outputBias, hiddenLayer, inputLayer
    inputLayer = flattenBoard()
    hiddenLayer = [0] * 20
    weights =  []
    for x in range(len(hiddenLayer)):
        nodesList = [random.uniform(-0.01,0.01) for y in range(len(inputLayer))]
        weights.append(nodesList)
    bias = [0] * 20

    outputWeights = []
    outputBias = [0] * 7
    outputLayer = [0] * 7
    for x in range(7):
        nodesList = [random.uniform(-0.01,0.01) for y in range(len(hiddenLayer))]
        outputWeights.append(nodesList)

#----- IF MATH DOES NOT EXIST (FILE)

learningRate = 0.01
def learnFromGame():
    for x in range(len(memoryMoves)):
        boardState = memoryBoards[x]
        move = memoryMoves[x]
        for y in range(len(hiddenLayer)):
            calculation = 0
            for i in range(len(boardState)):
                result = boardState[i] * weights[y][i]
                calculation += result
            calculation += bias[y]
            hiddenLayer[y] = calculation
    for h in range(len(hiddenLayer)):
        outputWeights[move][h] += learningRate * reward * hiddenLayer[h]
    outputBias[move] += learningRate * reward


def flattenBoard():
    flattenedBoard = []
    for row in board:
        for item in row:
            if item == 0:
                flattenedBoard.append(0)
            elif item == 1:
                flattenedBoard.append(1)
            elif item == 2:
                flattenedBoard.append(2)
    return np.array(flattenedBoard)

#winCheck 
def winCheck():
    #horizontalwin
    for row in range(len(board)):
        for column in range(4):
            if (board[row][column] == 1 and board[row][column+1] == 1 and board[row][column+2] == 1 and board[row][column+3] == 1) or (board[row][column] == 2 and board[row][column+1] == 2 and board[row][column+2] == 2 and board[row][column+3] == 2):
                return "win"
    #verticalwin
    for row in range(3):
        for column in range(7):
            if (board[row][column] == 1 and board[row+1][column] == 1 and board[row+2][column] == 1 and board[row+3][column] == 1) or (board[row][column] == 2 and board[row+1][column] == 2 and board[row+2][column] == 2 and board[row+3][column] == 2): 
                return "win"
    #diagonalwindown
    for row in range(3):
        for column in range(4):
            if (board[row][column] == 1 and board[row+1][column+1] == 1 and board[row+2][column+2] == 1 and board[row+3][column+3] == 1) or (board[row][column] == 2 and board[row+1][column+1] == 2 and board[row+2][column+2] == 2 and board[row+3][column+3] == 2): 
                return "win"
    #diagonalwinup
    for row in range(3,6):
        for column in range(4):
            if (board[row][column] == 1 and board[row-1][column+1] == 1 and board[row-2][column+2] == 1 and board[row-3][column+3] == 1) or (board[row][column] == 2 and board[row-1][column+1] == 2 and board[row-2][column+2] == 2 and board[row-3][column+3] == 2):
                return "win"
    #draw
    if not (board == 0).any():
        return "draw"

turn = 1

reward = 0




if os.path.exists("connect4.npz"):
    data = np.load("connect4.npz", allow_pickle=True)
    weights=data["weights"]
    bias=data["bias"]
    outputWeights=data["outputWeights"]
    outputBias=data["outputBias"]
else:
    initMath()
#game
while True:
    clearBoard()
    displayBoard()
    print("Input column (1-7)")

    if turn == 1:
        chosenColumn = getPlayerMove()
        while getLowestRow(chosenColumn) is None:
            chosenColumn = getPlayerMove()
        row = getLowestRow(chosenColumn)
        board[row][chosenColumn -1 ] = 1
        displayBoard()
        checkWin = winCheck()
        if checkWin == "draw":
            print("Game is a draw")
            reward = 0
            break
        elif checkWin == "win":
            print("Player 1 wins!")
            reward = -1
            break
        else:
            turn = 2
    elif turn == 2:
        chosenColumn = getAIMove()
        while getLowestRow(chosenColumn) is None:
            chosenColumn = getAIMove()
        row = getLowestRow(chosenColumn)
        board[row][chosenColumn -1] = 2
        displayBoard()
        checkWin = winCheck()
        if checkWin == "draw":
            print("Game is a draw")
            reward = 0
            break
        elif checkWin == "win":
            reward = 1
            print("AI wins!")
            break
        else:
            turn = 1
    else:
        print("Error in 'turn' variable")
        
learnFromGame()
np.savez(
    "connect4.npz",
    weights=weights,
    bias=bias,
    outputWeights=outputWeights,
    outputBias=outputBias
)
