from enum import Enum

from player_quoridor import PlayerQuoridor

from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math
import random
from fonctions.binou_chooser import Chooser
from typing import Optional
#K = 6
K_MOVES, K_WALLS = 4, 30
class GamePhase(Enum):
    OPENING = "opening"
    MIDGAME = "midgame"
    LATEGAME = "lategame"
def determine_game_phase(current_state: GameStateQuoridor, me: PlayerQuoridor, opponent: PlayerQuoridor)->GamePhase:
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

    # Lategame : quelqu'un est proche de l'arrivée, ou peu de murs restent
    if min_dist <= 4 or total_walls_left <= 14:
        return GamePhase.LATEGAME

    # Opening : tout le monde est loin de l'arrivée ET plusieurs murs sont encore disponibles
    if min_dist >= 6 and total_walls_left >= 17: 
        return GamePhase.OPENING


    # Sinon, on est entre les trois
    return GamePhase.MIDGAME
class Node:
    """
    A node in the Monte Carlo Tree Search (MCTS) tree.
    """
    def __init__(self, state: GameStateQuoridor, chooser: Chooser, me_id, parent: Optional["Node"] = None, action: Optional["Action"]=None):
        """
            Initialize the Node instance.
    
            Args:
                state (GameStateQuoridor): The current game state.
                chooser (Chooser): The action chooser and simulator for the player.
                me_id: The ID of the player using this node.
                parent (Optional[Node]): The parent node in the MCTS tree (default is None).
                action (Optional[Action]): The action that led to this node from the parent (default is None).
                """
        self.state = state
        self.chooser = chooser
        self.parent = parent
        self.action = action              # coup qui mène du parent à ce nœud
        self.me = self.state.players[0] if self.state.players[0].id == me_id else self.state.players[1]
        self.opponent = self.state.players[1] if self.state.players[0].id == me_id else self.state.players[0]
        self.children = {}  
        self._untried = None
        self.N = 0
        self.U = 0.0
        # joueur qui a joué l'action menant ici (pour le signe de U)
        self.player_just_moved = parent.state.active_player.get_id() if parent else None
        
    def _cut(self, actions):
        """
        Cuts the number of actions to a max of K_MOVES moves and K_WALLS walls, prioritizing moves over walls."""
        moves = [a for a in actions if a.data['type'] == 'move'][:K_MOVES]
        walls = [a for a in actions if a.data['type'] != 'move'][:K_WALLS]
        return moves + walls
    @property
    def _untried_actions(self):
        """
        Lazy initialization of untried actions. If the node is terminal, returns an empty list. Otherwise, generates candidate actions and stores them for future use.
        """
        if self._untried is None:
            self._untried = [] if self.state.is_done() else self._candidate_actions()
        return self._untried
    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def is_terminal(self):
        return self.state.is_done()

    def _candidate_actions(self):
        """
            Generates a list of candidate actions for the node, orders, filters, and limits them. Returns the cut list of actions to be considered.
        """
        actions = list(self.state.generate_possible_stateless_actions())
        ordered = list(self.chooser._order_actions(self.state, actions, self.state.active_player))
        filtered = list(self.chooser.delete_useless_walls(self.state, ordered))
        limited = list(self.chooser.square_grid(self.state, filtered, self.opponent, self.me))
        if self.parent is None:  
            minimaxed = list(self.chooser.filter_best_actions(self.state, limited)) if len(limited) > 30 else list(self.chooser.filter_best_actions(self.state, filtered))
            return minimaxed
        if len(limited) < 30:
            return self._cut(filtered)
        return self._cut(limited) or self._cut(filtered)

    
        


def ucb1(child: Node, c=0.2):
    """
        Selection condition (UBC1)
    """
    if child.N == 0:
        return float('inf')                      
    q = child.U / child.N                        
    explore = c * math.sqrt(math.log(child.parent.N) / child.N)
    return q + explore

def select(node :Node) :
    return max(node.children.values(), key=lambda ch: ucb1(ch))

def expand(node: Node):
    """Adds child to node
    """
    action = node._untried_actions.pop(0)
    new_state = node.state.apply_action(action)
    child = Node(new_state, node.chooser, node.me.get_id(), parent=node, action=action)
    node.children[action] = child
    return child

def simulate_with_random(node: Node, sim_player: Chooser, max_depth: int, epsilon = 0.1) :
    depth = 0
    current_state = node.state
    while not current_state.is_done() and depth < max_depth:

        if random.random() < epsilon:
            actions = tuple(current_state.generate_possible_stateless_actions())
            action = random.choice(actions)
        else:
            action = sim_player.compute_action(node.state, 1)
        current_state = current_state.apply_action(action)
        depth += 1
    if current_state.is_done():
        return current_state._opponent(current_state.active_player).get_id(), None
    cost = current_state._shortest_path(node.opponent) - current_state._shortest_path(node.me)
    return None, (1 / (1 + math.exp(-cost / 2)))

def backprop(node: Node, winner, me: PlayerQuoridor, value = None):
    while node is not None:
        node.N+=1
        if winner is not None:
            node.U += 1.0 if node.player_just_moved == winner else 0.0
        else:
            node.U += value if node.player_just_moved == me.get_id() else (1 - value)
        node = node.parent
def _depth(node):
    d = 0
    while node.parent:
        node, d = node.parent, d + 1
    return d


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
        

    def mcts(self, root_state: GameStateQuoridor, n: int, sim_agent: Chooser, me_id, sim_depth: int) -> Action:
        root = Node(root_state, sim_agent, me_id)
        for i in range(n):

            #sélection
            node = root
            while not node.is_terminal() and not node._untried_actions:
                """
                Descend in the tree until a node is found that is either terminal or has untried actions.
                """
                node = select(node)
            if node._untried_actions:
                node = expand(node)
            #Simulation
            winner, value = simulate_with_random(node, sim_agent, sim_depth)
            #Backpropagation
            backprop(node, winner, node.me, value)
        """
        #For loggings
        enfants = sorted(root.children.values(), key=lambda ch: -ch.N)
        log.debug(f"[RESULTAT] {len(enfants)} enfants étendus, {len(root._untried_actions)} candidats jamais essayés")
        for ch in enfants:
            q = ch.U / ch.N if ch.N else float('nan')
            log.debug(f"[RESULTAT] N={ch.N:4d} Q={q:.3f} action={ch.action}")  
        """    
        #Returns the action with the highest visit count 
        return max(root.children.values(), key=lambda ch: ch.N).action


    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use a Monte Carlo Tree Search to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by MCTS.
        """
        phase = determine_game_phase(current_state, current_state.active_player, current_state._opponent(current_state.active_player))
        my_id = self.get_id()
        chooser = Chooser(self.piece_type, self.goal_row, self.name)
        chooser.id = self.id
        "Si le joueur adverse n'a plus de mur ET que mon chemin est plus court que le sien, foncer devant"
        if not current_state.rep.remaining_walls[current_state._opponent(self)]:
            if current_state._shortest_path(current_state._opponent(self)) - current_state._shortest_path(self) > 0:
                me_path = chooser.bfs_shortest_path(current_state, self)
                next_square = me_path[1]
                shortcut_action = chooser._move_action_to(current_state, next_square)
                if shortcut_action is not None:
                    return shortcut_action
        if phase == GamePhase.LATEGAME:    
            best_action = self.mcts(current_state, 600, chooser, my_id, sim_depth = 0)
        elif phase == GamePhase.MIDGAME:
            best_action = self.mcts(current_state, 600, chooser, my_id, sim_depth = 0)
        else:
            best_action = chooser.compute_action(current_state, 1)
        return best_action