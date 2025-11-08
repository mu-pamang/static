"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread, Lock, Condition
from typing import Optional, Callable
import logging
import traceback
from .CountLatch import CountLatch

logger = logging.getLogger(__name__)


class CountLatch:
    """
    A counter that allows waiting until the count reaches zero.
    """
    
    def __init__(self, initial_count: int = 0):
        self._count = initial_count
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._interrupted = False
    
    def increment(self):
        """Increment the counter."""
        with self._lock:
            self._count += 1
    
    def decrement(self):
        """Decrement the counter and notify waiting threads if count reaches zero."""
        with self._lock:
            self._count -= 1
            if self._count <= 0:
                self._condition.notify_all()
    
    def await_zero(self, timeout: Optional[float] = None):
        """
        Wait until the counter reaches zero.
        
        Args:
            timeout: Maximum time to wait in seconds (None for infinite wait)
            
        Raises:
            InterruptedError: If the wait was interrupted
        """
        with self._lock:
            if timeout is None:
                while self._count > 0 and not self._interrupted:
                    self._condition.wait()
            else:
                while self._count > 0 and not self._interrupted:
                    if not self._condition.wait(timeout=timeout):
                        break
            
            if self._interrupted:
                raise InterruptedError("Wait was interrupted")
    
    def reset_and_interrupt(self):
        """Reset the counter to zero and interrupt all waiting threads."""
        with self._lock:
            self._count = 0
            self._interrupted = True
            self._condition.notify_all()
    
    def get_count(self) -> int:
        """Get the current count."""
        with self._lock:
            return self._count


class CountingThreadPoolExecutor(ThreadPoolExecutor):
    """
    A ThreadPoolExecutor which keeps track of the number of spawned tasks 
    to allow clients to await their completion.
    """
    
    def __init__(self, core_pool_size: int, maximum_pool_size: int, 
                 keep_alive_time: float, time_unit: str, work_queue: Queue):
        """
        Initialize the CountingThreadPoolExecutor.
        
        Args:
            core_pool_size: The number of threads to keep in the pool
            maximum_pool_size: The maximum number of threads to allow in the pool
            keep_alive_time: Keep alive time for threads
            time_unit: Time unit for keep_alive_time (not used in Python's ThreadPoolExecutor)
            work_queue: The queue to use for holding tasks (not directly used in Python's ThreadPoolExecutor)
        """
        # Python's ThreadPoolExecutor doesn't support all these parameters directly
        # We'll use max_workers as the main configuration
        super().__init__(max_workers=maximum_pool_size)
        
        self.num_running_tasks = CountLatch(0)
        self.exception: Optional[Exception] = None
        self._shutdown = False
        self._lock = Lock()
    
    def submit(self, fn: Callable, *args, **kwargs):
        """
        Submit a callable to be executed.
        
        Args:
            fn: The callable to execute
            *args: Positional arguments for the callable
            **kwargs: Keyword arguments for the callable
            
        Returns:
            A Future representing the pending execution
            
        Raises:
            RuntimeError: If the executor has been shut down
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new tasks after shutdown")
        
        try:
            self.num_running_tasks.increment()
            # Wrap the function to handle exceptions
            future = super().submit(self._wrapped_task, fn, *args, **kwargs)
            return future
        except Exception as ex:
            # If we were unable to submit the task, we may not count it!
            self.num_running_tasks.decrement()
            raise ex
    
    def _wrapped_task(self, fn: Callable, *args, **kwargs):
        """
        Wrapper for tasks to handle exceptions and decrement counter.
        
        Args:
            fn: The callable to execute
            *args: Positional arguments for the callable
            **kwargs: Keyword arguments for the callable
            
        Returns:
            The result of the callable
        """
        try:
            result = fn(*args, **kwargs)
            self.num_running_tasks.decrement()
            return result
        except Exception as t:
            # Store the exception
            self.exception = t
            logger.error(f"Worker thread execution failed: {str(t)}", 
                        exc_info=True)
            
            # Shutdown the executor
            self.shutdown(wait=False, cancel_futures=True)
            self.num_running_tasks.reset_and_interrupt()
            raise
    
    def execute(self, command: Callable):
        """
        Execute a command (for compatibility with Java's ThreadPoolExecutor interface).
        
        Args:
            command: The callable to execute
        """
        self.submit(command)
    
    def await_completion(self, timeout: Optional[float] = None):
        """
        Await the completion of all spawned tasks.
        
        Args:
            timeout: Maximum time to wait in seconds (None for infinite wait)
            
        Raises:
            InterruptedError: If the wait was interrupted
        """
        self.num_running_tasks.await_zero(timeout)
    
    def get_exception(self) -> Optional[Exception]:
        """
        Returns the exception thrown during task execution (if any).
        
        Returns:
            The exception that occurred, or None if no exception occurred
        """
        return self.exception
    
    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """
        Shutdown the executor.
        
        Args:
            wait: If True, wait for all tasks to complete
            cancel_futures: If True, cancel pending futures (Python 3.9+)
        """
        with self._lock:
            self._shutdown = True
        
        # Python 3.9+ supports cancel_futures parameter
        try:
            super().shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            # Fallback for Python < 3.9
            super().shutdown(wait=wait)
    
    def shutdown_now(self):
        """
        Immediately shutdown the executor (similar to Java's shutdownNow).
        Cancels all pending tasks and attempts to stop running tasks.
        """
        self.shutdown(wait=False, cancel_futures=True)


# For compatibility with the original Java time units
class TimeUnit:
    """Time unit constants for compatibility."""
    NANOSECONDS = 'nanoseconds'
    MICROSECONDS = 'microseconds'
    MILLISECONDS = 'milliseconds'
    SECONDS = 'seconds'
    MINUTES = 'minutes'
    HOURS = 'hours'
    DAYS = 'days'
    
    @staticmethod
    def to_seconds(value: float, unit: str) -> float:
        """
        Convert time value to seconds.
        
        Args:
            value: The time value
            unit: The time unit
            
        Returns:
            Time in seconds
        """
        conversions = {
            TimeUnit.NANOSECONDS: 1e-9,
            TimeUnit.MICROSECONDS: 1e-6,
            TimeUnit.MILLISECONDS: 1e-3,
            TimeUnit.SECONDS: 1.0,
            TimeUnit.MINUTES: 60.0,
            TimeUnit.HOURS: 3600.0,
            TimeUnit.DAYS: 86400.0,
        }
        return value * conversions.get(unit, 1.0)
    

"""\n===== ORIGINAL JAVA FOR REFERENCE =====\n/*******************************************************************************
 * Copyright (c) 2012 Eric Bodden.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Eric Bodden - initial API and implementation
 ******************************************************************************/
package heros.solver;

import heros.util.SootThreadGroup;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@link ThreadPoolExecutor} which keeps track of the number of spawned
 * tasks to allow clients to await their completion. 
 */
public class CountingThreadPoolExecutor extends ThreadPoolExecutor {
	
    protected static final Logger logger = LoggerFactory.getLogger(CountingThreadPoolExecutor.class);

    protected final CountLatch numRunningTasks = new CountLatch(0);
	
	protected volatile Throwable exception = null;

	public CountingThreadPoolExecutor(int corePoolSize, int maximumPoolSize, long keepAliveTime, TimeUnit unit,
			BlockingQueue<Runnable> workQueue) {
		super(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue, new ThreadFactory() {
			
			@Override
			public Thread newThread(Runnable r) {
				return new Thread(new SootThreadGroup(), r);
			}
		});
	}

	@Override
	public void execute(Runnable command) {
		try {
			numRunningTasks.increment();
			super.execute(command);
		}
		catch (RejectedExecutionException ex) {
			// If we were unable to submit the task, we may not count it!
			numRunningTasks.decrement();
			throw ex;
		}
	}
	
	@Override
	protected void afterExecute(Runnable r, Throwable t) {
		if(t!=null) {
			exception = t;
			logger.error("Worker thread execution failed: " + t.getMessage(), t);
			
			shutdownNow();
            numRunningTasks.resetAndInterrupt();
		}
		else {
			numRunningTasks.decrement();
		}
		super.afterExecute(r, t);
	}

	/**
	 * Awaits the completion of all spawned tasks.
	 */
	public void awaitCompletion() throws InterruptedException {
		numRunningTasks.awaitZero();
	}
	
	/**
	 * Awaits the completion of all spawned tasks.
	 */
	public void awaitCompletion(long timeout, TimeUnit unit) throws InterruptedException {
		numRunningTasks.awaitZero(timeout, unit);
	}
	
	/**
	 * Returns the exception thrown during task execution (if any).
	 */
	public Throwable getException() {
		return exception;
	}

}\n"""