import time
import math
import threading
import copy
import numpy as np
class PlayerAgent:
    def __init__(self, player_id):
        self.player_id = player_id  # play turn
        self.calculation_time = 4  # maximum calculation time
        self.max_actions = 1000  # max simulated actions
        self.confident = 1.96
        self.equivalence = 1000 # calc beta
        self.plays = {}
        self.wins = {}
        self.max_depth = 1
        self.count = 1
        self.simulations = 0
    def choose_move(self, game):
        print(self.count)
        self.count += 1
        self.simulations = 0
        # Get an array of legal moves from your current position.
        legal_moves = game.get_legal_moves(self.player_id)

        my_board = game.board
        # if there is only one place to move
        if len(legal_moves) == 1:
            # print(legal_moves)
            return legal_moves[0]
        self.plays = {}
        self.wins = {}

        threads = []
        nloops = 20
        board_copy = copy.deepcopy(my_board)

        for i in range(nloops):
            t = threading.Thread(target=self.start_thread, args=(board_copy,self.player_id,game))
            threads.append(t)

        for i in range(nloops):
            threads[i].start()

        for i in range(nloops):
            threads[i].join()

        # print("total simulations:", self.simulations)
        move = self.select_one_move(my_board,game)
        # print("maximum depth searched:", self.max_depth)
        # print("I choose:",move)

        return move
    def start_thread(self,board_copy,current_player,game):
        begin = time.time()
        # simulation = 0
        while time.time() - begin < self.calculation_time:

            self.start_simulation(board_copy, current_player, game)
            # simulation+=1
        # self.simulations +=simulation
    def start_simulation(self, board_copy, current_player, game):

        available_moves = game.get_legal_moves(current_player, board_copy)
        move = available_moves[0]
        visited_states = set()
        winner = -1
        expand = True
        for turn in range(1, self.max_actions + 1):
            if len(available_moves)>0:
                if all(self.plays.get((current_player, str(move))) for move in available_moves):
                    log_total = math.log(sum(self.plays[(current_player, str(move))] for move in available_moves))

                    value, move = max(((self.wins[(current_player, str(move))] / self.plays[(current_player, str(move))]) + np.sqrt(
                        self.confident * log_total / self.plays[(current_player,str(move))]), move) for move in available_moves)

                    board_copy = game.examine_move(current_player, move, board_copy)
                else:
                    adjacents = []
                    move = available_moves[0]

                    board_copy = game.examine_move(current_player, move, board_copy)

                if expand and (current_player, str(move)) not in self.plays:
                    expand = False
                    self.plays[(current_player, str(move))] = 0
                    self.wins[(current_player, str(move))] = 0

                    if turn > self.max_depth:
                        self.max_depth = turn

                visited_states.add((current_player,str(move)))
                win, winner = self.has_a_winner(board_copy, current_player, game)
                if win:
                    break
                current_player = 3 if current_player == 1 else 1
                available_moves = game.get_legal_moves(current_player, board_copy)
            else:
                winner = 3 if current_player == 1 else 1
        for player,move in visited_states:
            if (player,move) in self.plays:
                self.plays[(player,move)]+=1
                if player==winner:
                    self.wins[(player,move)]+=1


    def select_one_move(self,board,game):
        percent_wins, move = max((self.wins.get((self.player_id,str(move)),0)/self.plays.get((self.player_id,str(move)),1),move) for move in game.get_legal_moves(self.player_id,board))
        return move

    def has_a_winner(self, board, current_player, game):
        legal_moves = game.get_legal_moves(current_player, board)
        if len(legal_moves) == 0:
            if current_player == 1:
                return True, 3
            else:
                return True, 1
        else:
            return False, -1
            # Choose an action to take based on the algorithm you
            # decide to implement. This method should return
            # one of the items in the 'legal_moves' array.
