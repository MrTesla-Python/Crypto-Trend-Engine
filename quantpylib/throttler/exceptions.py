import json

class HTTPException(Exception):
    """
    An exception class for HTTP errors, capturing the status code, error message, and response headers.

    Args:
        status_code (int): The HTTP status code of the error.
        message (str): The error message.
        headers (dict): The response headers associated with the error.
    """

    def __init__(self, status_code, message, headers, cargs=None):
        self.status_code = status_code
        self.message = message
        self.headers = headers
        self.cargs = {} if cargs is None else cargs

    def __repr__(self):
        return f'status{self.status_code} :: {self.message}\n{self.headers}' + (f'\n{json.dumps(self.cargs)}' if self.cargs else '')

    __str__ = __repr__
