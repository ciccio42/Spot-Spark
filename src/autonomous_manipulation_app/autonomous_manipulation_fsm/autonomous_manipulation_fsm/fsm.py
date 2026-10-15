from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    DETECT = auto()
    NAVIGATE_TO_PICK = auto()
    PICK = auto()
    NAVIGATE_TO_PLACE = auto()
    PLACE = auto()
    RETURN_TO_IDLE = auto()
    ERROR = auto()


# allowed transitions: state -> states reachable from it
TRANSITIONS = {
    State.IDLE:              {State.DETECT},
    State.DETECT:            {State.NAVIGATE_TO_PICK, State.PICK, State.DETECT, State.ERROR},
    State.NAVIGATE_TO_PICK:  {State.PICK, State.DETECT, State.ERROR},
    State.PICK:              {State.NAVIGATE_TO_PLACE, State.DETECT, State.ERROR},
    State.NAVIGATE_TO_PLACE: {State.PLACE, State.ERROR},
    State.PLACE:             {State.RETURN_TO_IDLE, State.ERROR},
    State.RETURN_TO_IDLE:    {State.IDLE, State.ERROR},
    State.ERROR:             {State.IDLE},
}


class FSM:
    def __init__(self, initial_state=State.IDLE, on_transition=None):
        self.state = initial_state
        # callback(old_state, new_state) called after every transition (e.g. for logging)
        self.on_transition = on_transition

    def transition(self, new_state):
        if new_state not in TRANSITIONS[self.state]:
            raise ValueError(f"Invalid transition {self.state.name} -> {new_state.name}")
        old_state, self.state = self.state, new_state
        if self.on_transition:
            self.on_transition(old_state, new_state)

    def force(self, new_state):
        # bypass the transition table, used to jump to ERROR from any state
        old_state, self.state = self.state, new_state
        if self.on_transition:
            self.on_transition(old_state, new_state)
