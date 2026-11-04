from collections import deque
from typing import List, Optional
from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action

def _move_action_to(current_state: GameStateQuoridor, destination: tuple[int, int]):
    for action in current_state._legal_moves():
        if tuple(action.data['destination']) == tuple(destination):
            return action
    return None
def bfs_shortest_path(current_state: GameStateQuoridor, player: PlayerQuoridor) -> Optional[List[tuple[int, int]]]:
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
        

def _wall_priority_from_path(current_state: GameStateQuoridor, path: list[tuple[int, int]]) -> dict:
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
    
def in_path(action: Action, path:List[tuple[int, int]]) -> bool:
    """
    Retourne un bool qui vérifie si une action est dans un chemin (peut être le chemin du joueur 1 ou 2)
    """
    destination = tuple(action.data['destination'])
    for p in path:
        if destination == tuple(p):
            return True
    return False
    

def _order_actions(current_state: GameStateQuoridor, actions: tuple, target_player: PlayerQuoridor) -> list:
    """
    Sorts actions based on their priority
    1. moves
    2. walls that block the opponent's shortest path
    3. other walls (for now)
    """
    player_path = bfs_shortest_path(current_state, target_player)
    wall_priority = _wall_priority_from_path(current_state, player_path)
    def sort_key(action):
        if action.data['type'] == 'move':
            return -1  # moves
        wall_tuple = (action.data['destination'][0], action.data['destination'][1], action.data['type']) #create a tuple for wall action, to compare with wall_priority
        return wall_priority.get(wall_tuple, float('inf'))  # Stray walls get the lowest priority (infinity), while walls that are part of wall_tuple are prioritized
    return sorted(actions, key=sort_key)

def delete_useless_walls(current_state: GameStateQuoridor, actions: tuple, opp:PlayerQuoridor) -> tuple:
    """
    Deletes from the tuple of actions the useless walls, which are the ones that are connected to the bottom of the board
    
    Args:
        current_state (GameStateQuoridor): The current game state.
        actions (tuple): The possible actions a player can make during a turn
    Returns:
    returns a reduced tuple of possible actions to consider    
    """
    opp_dist = current_state._shortest_path(opp)
    def is_useless(a):
            if a.data['type'] == 'move':
                return opp_dist == 1
            row, col = a.data['destination']
            return (row == 0 and a.data['type'] == 'horizontal') or (col == 0 and a.data['type'] == 'vertical')
    filtered = tuple(a for a in actions if not is_useless(a))
    return filtered if filtered else actions

def square_grid(current_state: GameStateQuoridor, actions: tuple, player1: PlayerQuoridor, player2: PlayerQuoridor)->tuple:
    """
    Creates a grid of possible useful moves
            
    Args:
        current_state (GameStateQuoridor): The current game state.
        actions (tuple): The possible actions a player can make during a turn
        me (PlayerQuoridor): my agent
        opponent (PlayerQuoridor): the opponent
    Returns:
    returns a reduced tuple of possible actions to consider, focusing on a grid on both players    
    """
    player1_path = bfs_shortest_path(current_state, player1)
    player2_path = bfs_shortest_path(current_state, player2)
    player1_position = current_state.rep.pawn_positions[player1.id]
    player2_position = current_state.rep.pawn_positions[player2.id]

    # "Modifié pour ne pas manquer d'opportunités de bloquer un chemin, donc on voit toujours toute la largeur du tableau avec une vision avant arrière de 5 cases"
    def inside_5x5(position, destination):
        center_row, center_col = position
        row, col = destination
        return (
        center_row - 4 <= row <= center_row + 4
        and
        center_col - 2 <= col <= center_col + 2
        )

    reduced_actions = tuple(a for a in actions if (inside_5x5(player1_position, a.data['destination']) 
                        or inside_5x5(player2_position, a.data['destination'])
                        or in_path(a, player1_path)
                        or in_path(a, player2_path)))
    return reduced_actions
 