import queue
from tracemalloc import start

from utils import _move_action_to, bfs_shortest_path, _wall_priority_from_path, in_path, _order_actions, delete_useless_walls, square_grid

from seahorse import game, player
from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError

from enum import Enum
import logging
from collections import deque
from typing import List, Optional

class GamePhase(Enum):
    OPENING = "opening"
    MIDGAME = "midgame"
    LATEGAME = "lategame"
    ENDGAME = "endgame"

class MyPlayer(PlayerQuoridor):
    """
    Player class for Quoridor game

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, goal_row: int=0, name: str = "bob", *args, **kwargs) -> None:
        """
        Initialize the PlayerQuoridor instance.

        Args:
            piece_type (str): Type of the player's game piece
            goal_row (int): The row the player wants to reach
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, goal_row, name)


    def advanced_heuristic(self, current_state: GameStateQuoridor, me: PlayerQuoridor, opp: PlayerQuoridor, phase: GamePhase) -> float:
        """
        Heuristic evaluation function for the current game state.
        I want to add an advanced heuristic using the remaining walls in the equation, not only the distance difference.
        The idea is that when I evaluate the heuristic, there is a difference of walls between myself and opp. If the
        difference is high, I can afford to put up more walls, and hence the difference of walls will enhance the total cost, vice versa.
        I can play with the weights to try to prioritize wall difference less (or more) depending on trial and error
        Perhaps using gamephase to choose heuristics
        Also, I want to prioritize the difference between the current difference in paths and the next move's difference in path

        Args:
            current_state (GameStateQuoridor): The current game state.
         """
        
        path_weight = 1.0
        wall_weight = 0.25

        path_cost = current_state._shortest_path(opp) - current_state._shortest_path(me)
        

        wall_cost = current_state.rep.remaining_walls[me] - current_state.rep.remaining_walls[opp]
        cost = path_cost
        match phase:
            case GamePhase.OPENING:
                # "Here we just want to get foward without overthinking stuff"
                cost = path_cost
                return cost  
            case GamePhase.MIDGAME:
                # "Here, the walls become increasingly important, we want to keep them to deliver crushing blows"
                # Update wall weight to not use as many walls
                wall_weight = 1.5
                cost = path_cost*path_weight + wall_cost*wall_weight
                return cost
            case GamePhase.LATEGAME:
                # Keep it simple stupid, it's ok to use up walls 
                cost = path_cost*path_weight 
                return cost
            case GamePhase.ENDGAME:
                # "Not many moves possible. At this point, the worth of walls is already shown in the original heuristic (hypothèse)"
                cost = path_cost*path_weight
                return cost
            case _:
                return adv_path_cost

        
        
    
    def _determine_game_phase(self, current_state: GameStateQuoridor, me: PlayerQuoridor, opponent: PlayerQuoridor, action_lenght:int)->GamePhase:
        """
        Détermine la phase de la partie selon la distance restante des joueurs
        et le nombre de murs encore disponibles.
        """
        my_dist = current_state._shortest_path(me)
        opp_dist = current_state._shortest_path(opponent)
        min_dist = min(my_dist, opp_dist)

        my_walls = current_state.rep.remaining_walls[me.id]
        opp_walls = current_state.rep.remaining_walls[opponent.id]
        total_walls_left = my_walls + opp_walls
        # Endgame : il après avoir enlevé les murs, il reste moins de 25 coups légaux
        if action_lenght < 25 :
            return GamePhase.ENDGAME
        # Lategame : quelqu'un est proche de l'arrivée, ou peu de murs restent
        if min_dist <= 4 or total_walls_left <= 7:
            return GamePhase.LATEGAME

        # Opening : tout le monde est loin de l'arrivée ET plusieurs murs sont encore disponibles
        if min_dist >= 6 and total_walls_left >= 17: 
            return GamePhase.OPENING
        # Sinon, on est entre les trois
        return GamePhase.MIDGAME
    
    def minimax(self, current_state: GameStateQuoridor, depth: int, maximizing_player: bool, alpha: float, beta: float, phase: GamePhase) -> float:
        """
        Minimax algorithm to evaluate the best possible move for the current player.

        Args:
            current_state (GameStateQuoridor): The current game state.
            depth (int): The depth of the minimax search tree.
            maximizing_player (bool): True if the current player is our player, False otherwise.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.
        Returns:
            returns a tuple of (best_cost, best_action)
            """
        opponent = current_state.players[1] if current_state.players[0].id == self.get_id() else current_state.players[0]
        me = current_state.players[0] if current_state.players[0].id == self.get_id() else current_state.players[1]
        if depth == 0 or current_state.is_done():
            cost = self.advanced_heuristic(current_state, me, opponent, phase)
            return cost, None

        actions = tuple(current_state.generate_possible_stateless_actions())

        current_player_goal_row = current_state.active_player.get_goal_row()
        if current_player_goal_row != 0:
            actions = tuple(reversed(actions))       
        if not actions:
            raise RuntimeError("No legal action available.")

        if maximizing_player:
            best_cost = float('-inf')
            #Trier actions
            ordered_actions = _order_actions(current_state, actions, opponent)
            logging.debug(f"[EVAL BEFORE] actions before ={len(ordered_actions)}")

            filtered_ordered_actions = delete_useless_walls(current_state, ordered_actions, opponent)
            logging.debug(f"[EVAL MID] actions without useless ={len(filtered_ordered_actions)}")

            limited_actions = square_grid(current_state, filtered_ordered_actions, opponent, me)
            logging.debug(f"[EVAL AFTER] actions after bigger grid ={len(limited_actions)}")

            for action in limited_actions:
                temp_state = current_state.apply_action(action)
                cost, _ = self.minimax(temp_state, depth - 1, False, alpha, beta, phase)  
                alpha = max(alpha, cost)  
                logging.debug(f"[MAX] depth={depth} action={action} cost={cost} alpha={alpha} beta={beta}")        
                if cost > best_cost:
                    best_cost = cost
                    best_action = action 
                if beta <= alpha:
                    logging.debug(f"[MAX-PRUNE] depth={depth} pruned after action={action} (alpha={alpha} >= beta={beta})")
                    break
            return best_cost, best_action
        else:
            best_cost = float('inf')
            #Trier actions
            ordered_actions = _order_actions(current_state, actions, me)
            logging.debug(f"[EVAL BEFORE] actions before ={len(ordered_actions)}")

            filtered_ordered_actions = delete_useless_walls(current_state, ordered_actions, opponent)
            logging.debug(f"[EVAL AFTER] actions without useless ={len(filtered_ordered_actions)}")

            limited_actions = square_grid(current_state, filtered_ordered_actions, opponent, me)
            logging.debug(f"[EVAL AFTER] actions after grid ={len(limited_actions)}")


            for action in limited_actions:
                temp_state = current_state.apply_action(action)
                cost, _= self.minimax(temp_state, depth - 1, True, alpha, beta, phase)
                logging.debug(f"[MIN] depth={depth} action={action} cost={cost} alpha={alpha} beta={beta}")
                beta = min(beta, cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
                if beta <= alpha:
                    logging.debug(f"[MIN-PRUNE] depth={depth} pruned after action={action} (beta={beta} <= alpha={alpha})")
                    break
            return best_cost, best_action

        
    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        #"initiate best_action and best cost"
        best_cost = float('-inf')
        best_action = None

        # Determine which id is whose for future use
        opponent = current_state.players[1] if current_state.players[0].id == self.get_id() else current_state.players[0]
        me = current_state.players[0] if current_state.players[0].id == self.get_id() else current_state.players[1]

        # Calculations for game phase
        phase_action = tuple(current_state.generate_possible_stateless_actions())
        phase_action_filtered = delete_useless_walls(current_state, phase_action, opponent)        
        limited_phase_actions = square_grid(current_state, phase_action_filtered, opponent, me)
        # logging.debug(f"Lenght of actions for game state eval={len(limited_phase_actions)}")
        phase = self._determine_game_phase(current_state, me, opponent, len(limited_phase_actions))

        depth_by_phase = {
        GamePhase.OPENING: 1,
        GamePhase.MIDGAME: 2,
        GamePhase.LATEGAME: 3,
        GamePhase.ENDGAME: 3
        }
        # Determines the depth in which minimax will be executed
        depth = depth_by_phase[phase]
        # Maximizing player is me (the active player)
        maximizing_player = self.get_id() == current_state.active_player.id

        # Initializes alpha and beta for pruning
        alpha = float('-inf')
        beta = float('inf')

        # Exceptions to help with time limits

        # "Si le joueur adverse n'a plus de mur ET que mon chemin est plus court que le sien, foncer devant"
        if not current_state.rep.remaining_walls[current_state._opponent(self)]:
            if current_state._shortest_path(current_state._opponent(self)) - current_state._shortest_path(self) > 0:
                me_path = bfs_shortest_path(current_state, self)
                next_square = me_path[1]
                shortcut_action = _move_action_to(current_state, next_square)
                if shortcut_action is not None:
                    logging.debug(f"[SHORTCUT ACTION] type={shortcut_action.data['type']} x={shortcut_action.data['destination'][0]} y={shortcut_action.data['destination'][1]}")
                    return shortcut_action
        logging.debug(f"Remaining time={remaining_time}")
        # Si je n'ai pas de mur, augmenter la profondeur, à moins que je n'ai plus de temps
        if not current_state.rep.remaining_walls[current_state.active_player]:
            phase = GamePhase.ENDGAME
            if remaining_time > 45:
                depth = 5
            else:
                depth = 3
        if remaining_time < 12:
            depth = 1

        # Appelle la fonction minimax et retourne l'action calculée
        best_cost, best_action = self.minimax(current_state, depth, maximizing_player, alpha, beta, phase)
        logging.debug(f"[RESULTING MOVE] depth={depth} action={best_action} cost={best_cost} alpha={alpha} beta={beta}")
        return best_action