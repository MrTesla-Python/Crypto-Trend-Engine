import time
import threading
import collections 

from datetime import datetime
from threading import Lock
from threading import Condition

class RateSemaphore():
    """
    A semaphore implementation that limits the access to method/requests (known as transactions) based on available credits, and manages
    credit-debit accounting of a common resource pool. Functions to be executed are submitted to the `RateSemaphore.transact` 
    method along with the number of credits it consumes and the time it takes for those credits to be refunded to the resource
    pool. Can be used to throttle API requests, database operations and so on. In most cases, the user would only need to interact with the 
    `RateSemaphore.transact` method.
    """

    def __init__(self, credits=1):
        """
        Initialize the `RateSemaphore` with an initial number of credits.

        Args:
            credits (float, optional): The initial number of credits the semaphore starts with. Defaults to 1.
        
        Raises:
            ValueError: If `credits` is less than 0.
        """
        if credits < 0:
            raise ValueError("semaphore initial credits must be >= 0")
        self._cond = Condition(Lock())
        self._credits = credits

    def __repr__(self):
        return f"Credit Semaphore, has credits {self._credits}."

    def transact(self, lambda_func, credits, refund_time, transaction_id=None, verbose=False):
        """
        Execute a parameter-less function using the semaphore for synchronization, using the `RateSemaphore.acquire` and
        `RateSemaphore.release` methods for proper synchronization. The refund mechanism is scheduled in a worker 
        thread and the transact method returns without waiting for credits to be refunded.
        Failed transactions (raised Exceptions) consume and refund credit in the same way as successful transactions.

        Args:
            lambda_func (callable): The function to execute as part of the transaction. Should not take in any parameters.
            credits (float): The number of credits required for the transaction.
            refund_time (float): The time in seconds after which the credits should be refunded.
            transaction_id (str, optional): Identifier for the transaction. Defaults to None.
            verbose (bool, optional): Whether to print verbose transaction information. Defaults to False.

        Returns:
            Any: The result of the transaction function.

        Raises:
            Exception: If the transaction function raises any exception.

        Notes:
            Any function `func` that takes in `*args`, `**kwargs` can easily be used with 
            the semaphore by passing in `lambda_func = lambda : func(*args,**kwargs)`.
        """
        if verbose: 
            print(f"{datetime.now()}:: TXN {transaction_id} acquiring CreditSemaphore")
        self.acquire(credits)

        if verbose: 
            print(f'{datetime.now()}:: TXN {transaction_id} entered CreditSemaphore...')
        try:
            result = lambda_func()
        finally:
            thread = threading.Thread(target=self._refund_later, args=(credits, refund_time), daemon=True)
            thread.start()
        
        if verbose: 
            print(f'{datetime.now()}:: TXN {transaction_id} exits CreditSemaphore, schedule refund in {refund_time}...')
        return result

    def _refund_later(self, credits, after_time):
        assert(after_time >= 0)
        time.sleep(after_time)
        self.release(credits)
        return

    def acquire(self, require_credits):
        """
        Acquire the semaphore, decrementing the resource pool by specified number of credits.
        If the existing resource pool is larger than credits required, decrement the credits
        and return immediately. If there is not enough credits on entry, block, wait until 
        some other thread has called release() until enough resources are freed up.
        This is done with proper interlocking so that if multiple acquire() calls are blocked, 
        release() will wake exactly one of them up. The implementation may pick one at random, 
        so the order in which blocked threads are awakened should not be relied 
        on and is OS-scheduler dependent.

        Args:
            require_credits (float): The number of credits required to enter the semaphore.

        Returns:
            bool: True when the credits were successfully acquired.

        """
        rc = False
        with self._cond:
            while self._credits < require_credits:
                self._cond.wait(timeout=None)
            else:
                self._credits -= require_credits
                rc = True
        return rc

    def release(self, taken_credits):
        """
        Refund the specified number of credits.

        Args:
            taken_credits (float): The number of credits to release.
        """
        with self._cond:
            self._credits += taken_credits
            self._cond.notify_all()
        return

import asyncio
import functools
class AsyncRateSemaphore():
    """
    An asynchronous semaphore implementation that limits access to coroutines (known as transactions) based on available credits,
    and manages credit-debit accounting of a common resource pool. Coroutines to be executed are submitted to the `AsyncRateSemaphore.transact` 
    method along with the number of credits it consumes and the time it takes for those credits to be refunded to the resource pool. 
    Can be used to throttle asynchronous work such as non-blocking API requests or database operations. In most cases, users would only
    need to interact with the `AsyncRateSemaphore.transact` method.
    """

    def __init__(self, credits=1, greedy_entry=False, greedy_exit=True):
        """
        Initialize the `AsyncRateSemaphore` with an initial number of credits.

        Args:
            credits (float, optional): The initial number of credits the semaphore starts with. Defaults to 1.
            greedy_entry (bool, optional): Determines whether to enforce a FIFO or greedy policy for pending coroutines
                                            on semaphore entry. Defaults to False.
            greedy_exit (bool, optional): Determines whether to enforce a FIFO or greedy policy for waking up pending coroutines
                                            on credit refund. Defaults to True.

        Raises:
            ValueError: If `credits` is less than 0.
        """
        if credits < 0:
            raise ValueError("Semaphore credit value must be >= 0")
        self._waiters = None
        self._credits = credits
        self._future_costs = {}
        self._future_refunds = {}
        self.greedy_entry = greedy_entry
        self.greedy_exit = greedy_exit

    def __repr__(self):
        return f"has credits {self._credits} and waiter count of {0 if not self._waiters else len(self._waiters)}."

    async def transact(self, coroutine, credits, refund_time, transaction_id=None, verbose=False):
        """
        Execute an asynchronous coroutine object using the semaphore for synchronization, utilizing the `AsyncRateSemaphore.acquire` and
        `AsyncRateSemaphore.release` methods for proper synchronization. The refund mechanism is scheduled on the running event loop
        and the transact method returns without waiting for credits to be refunded.
        Failed transactions (raised Exceptions) consume and refund credit in the same way as successful transactions.
        
        Args:
            coroutine (coroutine function): The coroutine to execute as part of the transaction.
            credits (float): The number of credits required for the transaction.
            refund_time (float): The time in seconds after which the credits should be refunded.
            transaction_id (str, optional): Identifier for the transaction. Defaults to None.
            verbose (bool, optional): Whether to print verbose transaction information. Defaults to False.

        Returns:
            Any: The result of the executed coroutine.

        Raises:
            Exception: If the coroutine raises any exception.

        Notes:
            The argument `coroutine` should NOT be a `asyncio.Task` object, since any `await` statements will trigger 
            its execution on the event loop before the semaphore is acquired.
        """
        assert(asyncio.iscoroutine(coroutine))

        if verbose: 
            print(f"{datetime.now()}:: TXN {transaction_id} acquiring CreditSemaphore")
        await self.acquire(credits, refund_time)

        if verbose: 
            print(f'{datetime.now()}:: TXN {transaction_id} entered CreditSemaphore...')
        try:
            result = await coroutine
        finally:
            self._refund_later(credits, refund_time)
        
        if verbose: 
            print(f'{datetime.now()}:: TXN {transaction_id} exits CreditSemaphore, schedule refund in {refund_time}...')
        return result

    def _refund_later(self, credits, after_time):
        """
        Schedule a refund of credits after a specified time.

        Args:
            credits (float): The number of credits to refund.
            after_time (float): The time in seconds after which the credits should be refunded.
        """
        assert(after_time >= 0)
        if after_time == 0:
            self.release(credits)
        else:
            asyncio.get_running_loop().call_later(
                after_time,
                functools.partial(self.release, credits)
            )
        return

    def _locked(self, require_credits):
        """
        Check if the semaphore is locked based on the required number of credits and waiters.

        Args:
            require_credits (float): The number of credits required.

        Returns:
            bool: True if the semaphore is locked, False otherwise.
        """
        respect_fifo = any(not w.cancelled() for w in (self._waiters or ()))
        return self._credits < require_credits or (not self.greedy_entry and respect_fifo)

    async def acquire(self, require_credits, refund_time):
        """
        Acquire the semaphore, decrementing the resource pool by specified number of credits.
        If the existing resource pool is larger than credits required, decrement the credits
        and return immediately. If there is not enough credits on entry, do a non-blocking
        wait until enough resources are freed up. Additionally, if `greedy_entry=False`, then 
        the executing transaction will wait behind the earlier pending transactions regardless of the resource 
        pool availability.

        Args:
            require_credits (float): The number of credits required to enter the semaphore.
            refund_time (float): The time in seconds after which the credits should be refunded.

        Returns:
            bool: True when the credits were successfully acquired.
        """
        if not self._locked(require_credits):
            self._credits -= require_credits
            return True

        if self._waiters is None:
            self._waiters = collections.deque()
        
        fut = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        self._future_costs[fut] = require_credits
        self._future_refunds[fut] = refund_time

        try:
            try:
                await fut
            finally:
                del self._future_costs[fut]
                del self._future_refunds[fut]
                self._waiters.remove(fut)

        except Exception:
            if not fut.cancelled():
                self._refund_later(self._future_costs[fut], self._future_refunds[fut])
                del self._future_costs[fut]
                del self._future_refunds[fut]
            raise

        self._wake_up_next()
        return True

    def release(self, taken_credits):
        """
        Refund the specified number of credits and wake up pending transactions that 
        are able to execute on the state of the resource pool.
        Additionally, if `greedy_exit=False`, then the number of pending transactions 
        woken up will respect the FIFO order until the resource pool is insufficient for the
        earliest transaction.

        Args:
            taken_credits (float): The number of credits to release.
        """
        self._credits += taken_credits
        self._wake_up_next()

    def _wake_up_next(self):
        """
        Wake up the next coroutine waiting on the semaphore if credits are available.
        """
        if not self._waiters:
            return

        for fut in self._waiters:
            if not fut.done() and self._credits >= self._future_costs[fut]:
                self._credits -= self._future_costs[fut]
                fut.set_result(True)
            elif not self.greedy_exit: break
        return
    