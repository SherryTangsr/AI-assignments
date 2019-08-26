# Import any libraries you might need to develop our agent.
import numpy as np
import matplotlib
import time
import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from IPython.display import HTML, clear_output
import copy


# TODO: Implement your agent here by modifying the 'choose_move' function.
# Do not change the instantiation function or any of the function signatures.
# TODO: Implement your agent here.
import math
import threading
import random
class PlayerAgent:
    def __init__(self, player_id):
        self.player_id = player_id  # play turn
        self.calculation_time = 4  # maximum calculation time
        self.max_actions = 500  # max simulated actions
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
        self.plays_rave = {}  # key:(move, state), value:visited times

        self.wins_rave = {}  # key:(move, state), value:{player: win times}
        threads = []
        nloops = 10
        board_copy = copy.deepcopy(my_board)
        # 完成所有线程分配，并不立即开始执行
        for i in range(nloops):
            t = threading.Thread(target=self.start_thread, args=(board_copy,self.player_id,game))
            threads.append(t)
        # 开始调用start方法，同时开始所有线程
        for i in range(nloops):
            threads[i].start()
        # join方法：主线程等待所有子线程执行完成，再执行主线程接下来的操作。
        for i in range(nloops):
            threads[i].join()

        print("total simulations:", self.simulations)
        move = self.select_one_move(my_board,game)
        print("maximum depth searched:", self.max_depth)
        print("I choose:",move)

        return move

    def get_board_state(self,board):
        return tuple(tuple(line) for line in board)
    def start_thread(self,board_copy,current_player,game):
        begin = time.time()
        simulation = 0
        while time.time() - begin < self.calculation_time:

            self.start_simulation(board_copy, current_player, game)
            simulation+=1
        self.simulations +=simulation
    def start_simulation(self, board_copy, current_player, game):

        available_moves = game.get_legal_moves(current_player, board_copy)
        available_moves = sorted(available_moves,key=lambda x:x[0])
        move = available_moves[0]
        visited_states = set()
        state_list = []
        winner = -1
        expand = True
        for turn in range(1, self.max_actions + 1):

            # print("turn",turn)
            # print((current_player, str(available_moves[0])))
            # print("len available moves",len(available_moves))
            # print("contains",plays.__contains__((current_player, str(available_moves[0]))))
            if len(available_moves)>0:
                state = self.get_board_state(board_copy)
                actions = [(tuple(m), current_player) for m in available_moves]
                action = ()
                move = []
                if all(self.plays.get((action,state)) for action in actions):
                    total = 0
                    for a,s in self.plays:
                        if s == state:
                            total +=self.plays.get((a,s))
                    beta = self.equivalence/(3*total+self.equivalence)

                    value,action = max(((1-beta)*(self.wins[(action,state)]/self.plays[(action,state)])+
                                       beta*(self.wins_rave[(action[0],state)][current_player]/self.plays_rave[(action[0],state)])+
                                       math.sqrt(self.confident*math.log(total)/self.plays[(action,state)]),action)
                                      for action in actions)
                    # value,move = max(((1-math.sqrt(self.equivalence/(3*self.plays_rave[str(move)]+self.equivalence)))*(self.wins[(current_player,str(move))]/self.plays[current_player,str(move)])+
                    #                   math.sqrt(self.equivalence/(3*self.plays_rave[str(move)]+self.equivalence))*(self.wins_rave[str(move)][current_player]/self.plays_rave[str(move)])+
                    #                   math.sqrt(self.confident*math.log(self.plays_rave[str(move)])/self.plays[(current_player,str(move))]),move)
                    #                  for move in available_moves)
                    move = list(action[0])
                    board_copy = game.examine_move(current_player,list(action[0]), board_copy)
                else:
                    # TODO: test
                    move = min(available_moves)
                    action = (tuple(move),current_player)
                    board_copy = game.examine_move(current_player, move, board_copy)

                # print("here move again is:",move)
                if expand and (action,state) not in self.plays:
                    # print("not in")
                    expand = False
                    self.plays[(action,state)] = 0
                    self.wins[(action,state)] = 0

                    if turn > self.max_depth:
                        self.max_depth = turn
                state_list.append((action,state))

                for (m,pp), s in state_list:
                    if (tuple(move),s) not in self.plays_rave:
                        self.plays_rave[(tuple(move),s)]=0
                        self.wins_rave[(tuple(move),s)] = {}
                        self.wins_rave[(tuple(move),s)][current_player] = 0
                visited_states.add((action,state))
                # print("visited_states",visited_states)
                win, winner = self.has_a_winner(board_copy, current_player, game)
                if win:
                    break
                current_player = 3 if current_player == 1 else 1
                available_moves = game.get_legal_moves(current_player, board_copy)
            else:
                winner = 3 if current_player == 1 else 1
        for i,((m_root,p),s_root) in enumerate(state_list):
            action = (m_root,p)
            if (action,s_root) in self.plays:
                self.plays[(action,s_root)]+=1
                if current_player == winner and current_player in action:
                    self.wins[(action,s_root)]+=1
            for ((m_sub,p),s_sub) in state_list[i:]:
                self.plays_rave[(m_sub,s_root)]+=1
                if winner in self.wins_rave[(m_sub,s_root)]:
                    self.wins_rave[(m_sub,s_root)][winner]+=1


    def select_one_move(self,board,game):
        # print("here!!!!!!!")
        percent_wins, move = max((self.wins.get(((tuple(move),self.player_id),self.get_board_state(board)),0)/self.plays.get(((tuple(move),self.player_id),self.get_board_state(board)),1),move) for move in game.get_legal_moves(self.player_id,board))
        # print("move-return:",move)

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


# This is an example of an agent which simply picks a move at random.
class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, game):
        # Get an array of legal moves from your current position.
        legal_moves = game.get_legal_moves(self.player_id)

        # Shuffle the legal moves and pick the first one. This is equivalent
        # to choosing a move randomly with no logic.
        np.random.shuffle(legal_moves)
        return legal_moves[0]

class PlayerAgent2:
    import copy
    def __init__(self, player_id):
        self.player_id = player_id
        # self.count = 1
    def choose_move(self, game):
        # print(self.count)
        # self.count+=1
        # Get an array of legal moves from your current position.
        legal_moves = game.get_legal_moves(self.player_id)
        my_board = game.board
        # print(legal_moves)
        # print(len(legal_moves))
        legal_moves_copy = copy.deepcopy(legal_moves)
        # Generate score dictionary
        score_dic = {}
        for each in legal_moves:
            score_dic[str(each)] = 0
        next_moves = []
        score = self.alpha_beta(1, float("-inf"), float("inf"), 10, my_board, score_dic, game)
        # print(score_dic)
        for each_move in legal_moves:
            if score_dic[str(each_move)] == score:
                # print("choose")
                next_moves.append(each_move)
        return next_moves[0]
        # Choose an action to take based on the algorithm you
        # decide to implement. This method should return
        # one of the items in the 'legal_moves' array.
    def alpha_beta(self, current_agent, alpha, beta, depth, my_board, score_dic, game):
        """
        :param current_agent: 1:self, 2:opponent
        :param alpha:
        :param beta:
        :param depth:
        :param legal_moves:
        """
        legal_moves = game.get_legal_moves(current_agent, my_board)
        if self.check_situation(legal_moves):  # no place to go
            if current_agent == 1:  # I lose the game
                # print("end-")
                return float("-inf")
            else:  # I win the game
                # print("end")
                return float("inf")
        if depth == 0:  # return to root node
            # print("0")
            return self.compute_score()  # calculate the score for next move
        if current_agent == 1:  # It is my turn
            for each_move in legal_moves:  # for each child
                # create next move
                next_board = game.examine_move(current_agent, each_move, my_board)
                # print("length of legal_moves:", len(legal_moves))
                value = self.alpha_beta(3, alpha, beta, depth - 1, next_board, score_dic, game)
                if value > alpha:
                    alpha = value
                    score_dic[str(each_move)] = alpha
                if alpha >= beta:
                    break
            return alpha
        elif current_agent == 3:  # It is not my turn
            for each_move in legal_moves:
                next_board = game.examine_move(current_agent, each_move, my_board)
                # print("length of legal_moves:", len(legal_moves))
                value = self.alpha_beta(1, alpha, beta, depth - 1, next_board, score_dic, game)
                if value < beta:
                    beta = value
                    score_dic[str(each_move)] = beta
                if alpha >= beta:
                    break
            return beta
    def check_situation(self, legal_moves):
        # Return true if there is no place to go
        return len(legal_moves) == 0
    def compute_score(self):
        return 0
    def get_best_move(self, legal_moves):
        value = self.compute_score()
# This handler will be used to time-out actions/games which take too long to compute.
# Note that this handler does not function on Windows based systems.
# def signal_handler(signum, frame):
#     raise TimeoutError("Timed out!")
# signal.signal(signal.SIGALRM, signal_handler)


class TronGame:
    def __init__(self, agent1_class, agent2_class, board_size, board_type):
        # Default board.
        if board_type == 'default':
            self.size = board_size
        # Board with obstacles and a fixed size of 10x10.
        elif board_type == 'obstacles':
            self.size = 10
        elif board_type == 'rocky':
            self.size = board_size
        else:
            raise ValueError('Invalid board type.')

        # Build the game board.
        self.board_type = board_type
        self.board = self.build_board(board_type)

        # Initialize the game state variables and set the values using the
        # 'reset_game()' method.
        self.reset_game()

        # Initialize our agents.
        self.agent1 = agent1_class(1)
        self.agent2 = agent2_class(3)

    def build_board(self, board_type):
        """
        This method takes a board_type: ['default', 'obstacles'] and returns a
        new board (NumPy matrix).
        """

        # Default board.
        if board_type == 'default':
            board = np.zeros((self.size, self.size))
            board[0, 0] = 1
            board[self.size - 1, self.size - 1] = 3

        # Board with obstacles and a fixed size of 10x10.
        elif board_type == 'obstacles':
            board = np.zeros((10, 10))
            board[1, 4] = 1
            board[8, 4] = 3
            board[3:7, 0:4] = 4
            board[3:7, 6:] = 4
        # Board with obstacles and a fixed size of 10x10.
        elif board_type == 'rocky':
            board = np.zeros((self.size, self.size))
            a = np.random.randint(2, size=(self.size, self.size))
            b = np.random.randint(2, size=(self.size, self.size))
            c = np.random.randint(2, size=(self.size, self.size))
            d = np.random.randint(2, size=(self.size, self.size))

            board = board + (a * b * c * d) * 4
            board[0, 0] = 1
            board[self.size - 1, self.size - 1] = 3


        else:
            raise ValueError('Invalid board type.')

        return board

    def reset_game(self):
        """
        Helper method which re-initializes the game state.
        """

        self.board = self.build_board(self.board_type)

    def get_player_position(self, player_id, board=None):
        """
        Helper method which finds the coordinate of the specified player ID
        on the board.
        """

        if board is None:
            board = self.board
        coords = np.asarray(board == player_id).nonzero()
        coords = np.stack((coords[0], coords[1]), 1)
        coords = np.reshape(coords, (-1, 2))
        return coords[0]

    def get_legal_moves(self, player, board=None):
        """
        This method returns a list of legal moves for a given player ID and
        board.
        """

        if board is None:
            board = self.board

        # Get the current player position and then check for all possible
        # legal moves.
        prev = self.get_player_position(player, board)
        moves = []

        # Up
        if (prev[0] != 0) and (board[prev[0] - 1, prev[1]] == 0):
            moves.append([prev[0] - 1, prev[1]])
        # Down
        if (prev[0] != self.size - 1) and (board[prev[0] + 1, prev[1]] == 0):
            moves.append([prev[0] + 1, prev[1]])
        # Left
        if (prev[1] != 0) and (board[prev[0], prev[1] - 1] == 0):
            moves.append([prev[0], prev[1] - 1])
        # Right
        if (prev[1] != self.size - 1) and (board[prev[0], prev[1] + 1] == 0):
            moves.append([prev[0], prev[1] + 1])

        return moves

    def examine_move(self, player, coordinate, board):
        board_clone = board.copy()
        prev = self.get_player_position(player, board_clone)
        board_clone[prev[0], prev[1]] = 4
        board_clone[coordinate[0], coordinate[1]] = player
        return board_clone

    @staticmethod
    def view_game(board_history):
        """
        This is a helper function which takes a board history
        (i.e., a list of board states) and creates an animation of the game
        as it progresses.
        """

        fig, ax = plt.subplots()
        colors = ['black', 'blue', 'pink', 'white', 'red', 'yellow']
        cmap = matplotlib.colors.ListedColormap(colors)
        bounds = np.linspace(0, 5, 6)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        matrix = ax.matshow(board_history[0], cmap=cmap, norm=norm)

        def update(i):
            matrix.set_array(board_history[i])
            return matrix

        ani = FuncAnimation(fig, update, frames=len(board_history), interval=100)
        plt.show()
        # return HTML(ani.to_html5_video())

    def play_series(self, num_games, debug=True):
        """
        This method plays a series of games between the two agents.

        It returns two objects: (i) a tuple which indicates the number of
        wins per player, and (ii) a history of the board state as the game
        progresses.
        """

        wins_player_1 = 0
        wins_player_2 = 0
        games = []
        for i in range(num_games):
            winning_player_id, board_history = self.__play_game(debug=debug)
            games.append(board_history)

            if winning_player_id == 1:
                wins_player_1 += 1
            elif winning_player_id == 2:
                wins_player_2 += 1
            else:
                raise ValueError('Invalid winning player ID.')

        print(f'Finished playing [{num_games}] games.')
        print(f'Player 1 won [{wins_player_1}] games and has a win-rate of [{wins_player_1 / num_games * 100}%].')
        print(f'Player 2 won [{wins_player_2}] games and has a win-rate of [{wins_player_2 / num_games * 100}%].')
        return (wins_player_1, wins_player_2), games

    def __apply_move(self, player, coordinate):
        """
        This private method moves a player ID to a new coordinate and obstructs
        the previously occupied tile.
        """

        prev_coord = self.get_player_position(player)

        self.board[prev_coord[0], prev_coord[1]] = 4
        self.board[coordinate[0], coordinate[1]] = player

    def __play_game(self, debug=True):
        """
        This private method plays a single game between the two agents. It
        returns the winning player ID as well as the history of the board
        as the game progresses.
        """

        # Reset the game.
        self.reset_game()
        board_history = []

        # Play the game until it's conclusion.
        while True:
            # ---------------------------------------
            # PLAYER 1's TURN
            # ---------------------------------------
            # Check legal moves.
            poss_moves = self.get_legal_moves(1)
            if not len(poss_moves):
                if debug:
                    print("Player 2 wins")
                winning_player_id = 2
                break

            # Compute and apply the chosen move.
            # signal.alarm(3)
            try:
                move = self.agent1.choose_move(self)
            except Exception:
                print("There was an error while choosing a move.")
                print("Player 2 wins")
                winning_player_id = 2
                break
            self.__apply_move(1, move)

            # Record keeping.
            board_history.append(np.array(self.board.copy()))
            if debug:
                print(self.board)
                time.sleep(.5)
                clear_output()

            # ---------------------------------------
            # PLAYER 2's TURN
            # ---------------------------------------
            # Check legal moves.
            poss_moves = self.get_legal_moves(3)
            if not len(poss_moves):
                if debug:
                    print("Player 1 wins")
                winning_player_id = 1
                break

            # Compute and apply the chosen move.
            # signal.alarm(3)
            try:
                move = self.agent2.choose_move(self)
            except Exception:
                print("There was an error while choosing a move.")
                print("Player 1 wins")
                winning_player_id = 1
                break
            self.__apply_move(3, move)

            # Record keeping.
            board_history.append(np.array(self.board.copy()))
            if debug:
                print(self.board)
                time.sleep(.5)
                clear_output()
        # signal.alarm(0)

        return winning_player_id, board_history


my_tron_game = TronGame(board_size=20,
                        agent1_class=PlayerAgent,
                        agent2_class=RandomAgent,
                        board_type='rocky')

(player1_wins, player2_wins), game_histories = my_tron_game.play_series(num_games=3, debug=False)
TronGame.view_game(game_histories[0])
TronGame.view_game(game_histories[1])
TronGame.view_game(game_histories[2])
TronGame.view_game(game_histories[3])
TronGame.view_game(game_histories[4])
TronGame.view_game(game_histories[5])
TronGame.view_game(game_histories[6])
TronGame.view_game(game_histories[7])
TronGame.view_game(game_histories[8])
TronGame.view_game(game_histories[9])
