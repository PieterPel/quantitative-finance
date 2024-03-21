class WrongDataFrameError(Exception):
    '''
    Error that signals that the given DataFrame isn't compatible
    '''


class UnfittedModelError(Exception):
    '''
    Error that is thrown if the model is not fitted yet
    '''
