import threading
import asyncio
import functools 

def wrap_in_thread(_func=None, *, costs, refund_in, attrname="rate_semaphore", daemon=False, verbose=True):
    """
    Decorator to wrap a synchronous instance-level method into a transaction, and that transaction into a thread. 
    The return value is a thread, that is not yet alive, and can be activated by invoking the 
    `threading.Thread.start` method.

    Args:
        _func (callable, optional): The function to be wrapped.
        costs (float): The number of credits required for the transaction.
        refund_in (float): The time in seconds after which the credits should be refunded.
        attrname (str, optional): The attribute name to access the RateSemaphore belonging to the object instance. Defaults to "rate_semaphore".
        daemon (bool, optional): Whether the thread should be a daemon thread. Defaults to False.
        verbose (bool, optional): Whether to print verbose transaction information. Defaults to True.

    Returns:
        (threading.Thread): The thread with the transaction as target function, where the transaction wraps the decorated function. Not yet alive.

    Raises:
        AssertionError: If the instance does not have the specified attribute or if costs is less than 0.
    """

    def wrapper(func):
        @functools.wraps(func)
        def helper(self, *args, **kwargs):    
            assert(hasattr(self, attrname))
            assert(costs >= 0)

            sem = getattr(self, attrname)
            lambda_func = lambda : func(self, *args, **kwargs) 
            thread = threading.Thread(
                target=sem.transact,
                kwargs={
                    "lambda_func":lambda_func, 
                    "credits":costs, 
                    "refund_time":refund_in, 
                    "transaction_id":{"fn": func.__name__, "args": args, "kwargs": kwargs}, 
                    "verbose":verbose
                },
                daemon=daemon
            ) 
            return thread
                      
        return helper

    if _func is None:
        return wrapper
    else:
        return wrapper(_func)

def consume_credits(_func=None, *, costs, refund_in, attrname="rate_semaphore", verbose=False):
    """
    Decorator to wrap an synchronous instance-level method as an synchronous transaction within a rate-limited semaphore.
    The return value is the return value of the decorated method.

    Args:
        _func (callable, optional): The method to be wrapped.
        costs (float): The number of credits required for the transaction.
        refund_in (float): The time in seconds after which the credits should be refunded.
        attrname (str, optional): The attribute name to access the RateSemaphore belonging to the object instance. Defaults to "rate_semaphore".
        verbose (bool, optional): Whether to print verbose transaction information. Defaults to True.

    Returns:
        (any): The return value of the decorated method.

    Raises:
        AssertionError: If the instance does not have the specified attribute or if costs is less than 0.
    """

    def wrapper(func):
        @functools.wraps(func)
        def helper(self, *args, **kwargs):    
            assert(hasattr(self, attrname))
            assert(costs >= 0)

            sem = getattr(self, attrname)
            lambda_func = lambda : func(self, *args, **kwargs) 
            return sem.transact(
                lambda_func=lambda_func,
                credits=costs,
                refund_time=refund_in,
                transaction_id={"fn": func.__name__, "args": args, "kwargs": kwargs}, 
                verbose=verbose
            )
                      
        return helper

    if _func is None:
        return wrapper
    else:
        return wrapper(_func)

def aconsume_credits(_func=None, *, costs, refund_in, attrname="rate_semaphore", timeout=0, verbose=False):
    """
    Decorator to wrap an asynchronous instance-level coroutine as an asynchronous transaction within a rate-limited semaphore.
    The return value is the return value of the decorated coroutine.

    Args:
        _func (coroutine, optional): The asynchronous coroutine to be wrapped.
        costs (float): The number of credits required for the transaction.
        refund_in (float): The time in seconds after which the credits should be refunded.
        attrname (str, optional): The attribute name to access the AsyncRateSemaphore belonging to the object instance. Defaults to "rate_semaphore".
        timeout (float, optional): The maximum time in seconds to wait for the transaction to complete. Defaults to 0, which waits indefinitely.
        verbose (bool, optional): Whether to print verbose transaction information. Defaults to True.

    Returns:
        (any): The return value of the decorated coroutine.

    Raises:
        AssertionError: If the instance does not have the specified attribute or if costs is less than 0.
        asyncio.TimeoutError: If wrapped transaction takes longer than the specified timeout.
    
    Notes:
        For a timed-out transaction that does not get a chance to run due to it sitting 
        behind other transactions that are opportunistically processed, 
        a `RuntimeWarning: Enable tracemalloc to get the object allocation traceback` 
        may be seen together with the `asyncio.TimeoutError`. This is because the transaction 
        contains the coroutine, and the transact is wrapped in an asyncio `task`. 
        If it is unable to enter the semaphore by timeout, 
        the coroutine is never awaited and complains when garbage collected. 
        If the timeout occurs after the coroutine has acquired the semaphore, 
        this warning will not be seen but the `asyncio.TimeoutError` will still 
        be thrown.
    """

    def wrapper(func):
        @functools.wraps(func)
        async def execute_transaction(self, *args, **kwargs):    
            assert(hasattr(self, attrname))
            assert(costs >= 0)

            sem = getattr(self, attrname)
            transaction = sem.transact(
                coroutine=func(self, *args, **kwargs), 
                credits=costs, 
                refund_time=refund_in, 
                transaction_id={"fn": func.__name__, "args": args, "kwargs": kwargs}, 
                verbose=verbose
            )

            if timeout == 0:
                result = await transaction
            else:
                result = await asyncio.wait_for(asyncio.create_task(transaction), timeout=timeout)
  
            return result
                      
        return execute_transaction

    if _func is None:
        return wrapper
    else:
        return wrapper(_func)