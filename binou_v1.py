from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError

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

    def minimax(self, current_state: GameStateQuoridor, depth: int, maximizing_player: bool) -> float:
        """
        Minimax algorithm to evaluate the best possible move for the current player.

        Args:
            current_state (GameStateQuoridor): The current game state.
            depth (int): The depth of the minimax search tree.
            maximizing_player (bool): True if the current player is our player, False otherwise.
        Returns:
            returns a tuple of (best_cost, best_action)
            """
        if depth == 0 or current_state.is_done():
            if current_state.players[0].id == self.get_id():
                me = current_state.players[0]
                opponent = current_state.players[1]
            else:
                me = current_state.players[1]
                opponent = current_state.players[0]
            cost = current_state._shortest_path(opponent) - current_state._shortest_path(me)
            return cost, None

        actions = tuple(current_state.generate_possible_stateless_actions())
        if not actions:
            raise RuntimeError("No legal action available.")

        if maximizing_player:
            best_cost = float('-inf')
            for action in actions:
                temp_state = current_state.apply_action(action)
                cost, _ = self.minimax(temp_state, depth - 1, False)
                if cost >= best_cost:
                    best_cost = cost
                    best_action = action
            return best_cost, best_action
        else:
            best_cost = float('inf')
            for action in actions:
                temp_state = current_state.apply_action(action)
                cost, _= self.minimax(temp_state, depth - 1, True)
                if cost <= best_cost:
                    best_cost = cost
                    best_action = action
            return best_cost, best_action

        
    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        "initiate best_action"
        best_cost = float('-inf')  # Initialize best_cost to a small value
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
        If computation is too heavy, we can limit the wall placements to only those that are next to the opponent/existing walls, and add alpha beta pruning later.
        """
        depth = 1
        maximizing_player = self.get_id() == current_state.active_player.id

        best_cost, best_action = self.minimax(current_state, depth, maximizing_player)
        return best_action