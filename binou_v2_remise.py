import queue
from tracemalloc import start

from seahorse import game, player
from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError


from collections import deque
from typing import List, Optional
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


    '''def heuristic(self, current_state: GameStateQuoridor) -> float:
        """Heuristic evaluation function for the current game state.

        Args:
            current_state (GameStateQuoridor): The current game state.
         """
        if current_state.players[0].id == self.get_id():
             me = current_state.players[0]
             opp = current_state.players[1]
        else:
             me = current_state.players[1]
             opp = current_state.players[0]
        me_row = current_state.get_rep().pawn_positions([me.id])[2]
        opp_row = current_state.get_rep().pawn_positions([opp.id])[2]
        distance_moi = self.get_goal_row() - me_row
        distance_opp = self.get_goal_row() - opp_row
        return distance_moi - distance_opp
        '''
    def _move_action_to(self, current_state: GameStateQuoridor, destination: tuple[int, int]):
        for action in current_state._legal_moves():
            if tuple(action.data['destination']) == tuple(destination):
                return action
        return None
    def bfs_shortest_path(self, current_state: GameStateQuoridor, player: PlayerQuoridor) -> Optional[List[tuple[int, int]]]:
        """
        Computes the shortest path between the player's pawn
        and its target row.

        The search ignores the opponent pawn and only considers the walls
        currently present on the board.

        Attributes
            player (Player): Player whose shortest path is computed.

        Returns
            Optional[List[tuple[int, int]]]: Ordered list of squares from the
            player's current position to a square on the goal row (inclusive),
            or None if no path exists.
        """

        start = current_state.rep.pawn_positions[player.id]
        queue = deque([start])
        visited = {start}
        predecessor = {start: None}

        while queue:

            position = queue.popleft()
            if position[0] == player.get_goal_row():
            # Reconstruction du chemin en remontant les prédécesseurs
                path = []
                node = position
                while node is not None:
                    path.append(node)
                    node = predecessor[node]
                path.reverse()
                return path

            for neighbour in current_state._reachable_neighbours(position):
                if neighbour not in visited:
                    visited.add(neighbour)
                    predecessor[neighbour] = position
                    queue.append(neighbour)

        return None
   # Concrètement, retourne une liste des cases qui correspondent au chemin le plus court.
        

    def _wall_priority_from_path(self, current_state: GameStateQuoridor, path: list[tuple[int, int]]) -> dict:
        """
        Associe chaque mur candidat (touchant une étape du chemin) à sa priorité.
        Priorité basse (proche de 0) = touche le chemin, à explorer en premier.
        """
        if path is None:
            return {}
        priority = {}
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            for wall in current_state._candidate_blocking_walls(start, end):
                orientation_str = wall.orientation.name.lower()
                wall_tuple = (wall.row, wall.col, orientation_str)
                if wall_tuple not in priority:
                    priority[wall_tuple] = i  # plus i est petit, plus c'est tôt dans le chemin
        return priority


    def _order_actions(self, current_state: GameStateQuoridor, actions: tuple, target_player: PlayerQuoridor) -> list:
        """
        Sorts actions based on their priority
        1. moves
        2. walls that block the opponent's shortest path
        3. other walls (for now)
        """
        player_path = self.bfs_shortest_path(current_state, target_player)
        wall_priority = self._wall_priority_from_path(current_state, player_path)

        def sort_key(action):
            if action.data['type'] == 'move':
                return -1  # moves
            wall_tuple = (action.data['destination'][0], action.data['destination'][1], action.data['type']) #create a tuple for wall action, to compare with wall_priority
            return wall_priority.get(wall_tuple, float('inf'))  # Stray walls get the lowest priority (infinity), while walls that are part of wall_tuple are prioritized

        return sorted(actions, key=sort_key)
    
    def delete_useless_walls(self, current_state: GameStateQuoridor, actions: tuple) -> tuple:
            """
            Deletes from the tuple of actions the useless walls, which are the ones that are connected to the bottom of the board
            
            Args:
                current_state (GameStateQuoridor): The current game state.
                actions (tuple): The possible actions a player can make during a turn
            Returns:
            returns a reduced tuple of possible actions to consider    
            """
            def is_useless(a):
                    if a.data['type'] == 'move':
                        return False
                    row, col = a.data['destination']
                    return (row == 0 and a.data['type'] == 'vertical') or (col == 0 and a.data['type'] == 'horizontal')
            return tuple(a for a in actions if not is_useless(a))
    
    def minimax(self, current_state: GameStateQuoridor, depth: int, maximizing_player: bool, alpha: float, beta: float) -> float:
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
            cost = current_state._shortest_path(opponent) - current_state._shortest_path(me)
            return cost, None
        
        actions = current_state.generate_possible_stateless_actions()
       
        if not actions:
            raise RuntimeError("No legal action available.")

        if maximizing_player:
            best_cost = float('-inf')
            #Trier actions
            ordered_actions = self._order_actions(current_state, actions, opponent)
            filtered_ordered_actions = self.delete_useless_walls(current_state, ordered_actions)
            for action in filtered_ordered_actions:
                temp_state = current_state.apply_action(action)
                cost, _ = self.minimax(temp_state, depth - 1, False, alpha, beta)  
                alpha = max(alpha, cost)     
                if cost > best_cost:
                    best_cost = cost
                    best_action = action 
                if beta <= alpha:

                    break
            return best_cost, best_action
        else:
            best_cost = float('inf')
            #Trier actions
            ordered_actions = self._order_actions(current_state, actions, me)
            filtered_ordered_actions = self.delete_useless_walls(current_state, ordered_actions)

            for action in filtered_ordered_actions:
                temp_state = current_state.apply_action(action)
                cost, _= self.minimax(temp_state, depth - 1, True, alpha, beta)
                beta = min(beta, cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_action = action
                if beta <= alpha:
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

        "initiate best_action and best cost"
        best_cost = float('-inf')
        best_action = None

        "The cost of an action represents the difference in shortest paths for the opponent and our agent. We want to minimize the cost for the player and maximize the cost for the opponent."
        "The cost is calculated as opponent.shortest - player.shortest, so a big value is good"
        """
        Every move has a cost of 1, the goal would be to always add more moves to the opponent and less moves to the player to reach the end row
        Hence two moves are possible : move or place a wall. If we place a wall, it needs to either apply or set up a >= 2 move deficit for the opponent
        since it costs a move to place a wall
        """

        """
        Meaning the minimax algorithm must include placing walls as a possible action.
        A fair assumption (?) is that placing walls next to the opponent or other walls are going to have a better outcome than other wall placements.
        This assumption cuts down computation time, but generate_possible_stateless_actions() already computes every possible wall placement.
        If computation is too heavy, we can limit the wall placements to only those that are next to the opponent/existing walls, and add alpha beta pruning later to search deeper.
        """
        depth = 2
        maximizing_player = self.get_id() == current_state.active_player.id
        alpha = float('-inf')
        beta = float('inf')
        if not current_state.rep.remaining_walls[current_state._opponent(self)]:
            if current_state._shortest_path(current_state._opponent(self)) - current_state._shortest_path(self) > 0:
                me_path = self.bfs_shortest_path(current_state, self)
                next_square = me_path[1]
                shortcut_action = self._move_action_to(current_state, next_square)
                if shortcut_action is not None:
                    return shortcut_action
        best_cost, best_action = self.minimax(current_state, depth, maximizing_player, alpha, beta)
        return best_action