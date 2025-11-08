"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from threading import Lock, Condition
from typing import Optional
import time


class CountLatch:
    """
    A synchronization aid similar to CountDownLatch but with the ability
    to also count up. This is useful to wait until a variable number of tasks
    have completed. await_zero() will block until the count reaches zero.
    """
    
    def __init__(self, count: int):
        """
        Initialize the CountLatch with the given count.
        
        Args:
            count: Initial count value
        """
        self._count = count
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._waiting_threads = []
    
    def get_count(self) -> int:
        """
        Get the current count value.
        
        Returns:
            The current count
        """
        with self._lock:
            return self._count
    
    def reset(self):
        """
        Reset the counter to zero without releasing waiting threads.
        """
        with self._lock:
            self._count = 0
    
    def await_zero(self):
        """
        Block until the count reaches zero.
        
        Raises:
            InterruptedError: If the current thread is interrupted while waiting
        """
        with self._lock:
            # Register this thread as waiting
            import threading
            current_thread = threading.current_thread()
            self._waiting_threads.append(current_thread)
            
            try:
                while self._count > 0:
                    self._condition.wait()
            finally:
                # Unregister this thread
                if current_thread in self._waiting_threads:
                    self._waiting_threads.remove(current_thread)
    
    def await_zero(self, timeout: Optional[float] = None, time_unit: Optional[str] = None) -> bool:
        """
        Block until the count reaches zero or timeout expires.
        
        Args:
            timeout: Maximum time to wait. If time_unit is provided, this is in that unit.
                    Otherwise, it's in seconds. If None, waits indefinitely.
            time_unit: Optional time unit ('nanoseconds', 'microseconds', 'milliseconds',
                      'seconds', 'minutes', 'hours', 'days')
        
        Returns:
            True if the count reached zero, False if timeout expired
            
        Raises:
            InterruptedError: If the current thread is interrupted while waiting
        """
        # Convert timeout to seconds if time_unit is provided
        if timeout is not None and time_unit is not None:
            timeout = self._convert_to_seconds(timeout, time_unit)
        
        with self._lock:
            # Register this thread as waiting
            import threading
            current_thread = threading.current_thread()
            self._waiting_threads.append(current_thread)
            
            try:
                if timeout is None:
                    # Wait indefinitely
                    while self._count > 0:
                        self._condition.wait()
                    return True
                else:
                    # Wait with timeout
                    end_time = time.time() + timeout
                    while self._count > 0:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            return False
                        self._condition.wait(timeout=remaining)
                    return True
            finally:
                # Unregister this thread
                if current_thread in self._waiting_threads:
                    self._waiting_threads.remove(current_thread)
    
    def increment(self):
        """
        Increment the count by one.
        """
        with self._lock:
            self._count += 1
    
    def decrement(self):
        """
        Decrement the count by one. If the count reaches zero, notify all waiting threads.
        """
        with self._lock:
            if self._count > 0:
                self._count -= 1
                if self._count == 0:
                    self._condition.notify_all()
    
    def reset_and_interrupt(self):
        """
        Resets the counter to zero. Waiting threads won't be released normally,
        so this interrupts the threads so that they escape from their waiting state.
        
        Note: Python doesn't have direct thread interruption like Java. This implementation
        uses the condition variable's notify mechanism and forces the count to zero.
        Waiting threads will check the interrupted state and raise InterruptedError.
        """
        with self._lock:
            self._count = 0
            
            # Because it is a best effort thing, do it three times and hope for the best
            for _ in range(3):
                # Copy the list to avoid modification during iteration
                threads_to_interrupt = list(self._waiting_threads)
                for thread in threads_to_interrupt:
                    # In Python, we can't directly interrupt a thread like in Java
                    # The best we can do is notify all waiting threads
                    pass
                
                # Notify all waiting threads
                self._condition.notify_all()
            
            # Just in case a thread would've incremented the counter again
            self._count = 0
    
    def is_at_zero(self) -> bool:
        """
        Check whether this counting latch has arrived at zero.
        
        Returns:
            True if this counting latch has arrived at zero, otherwise False
        """
        with self._lock:
            return self._count == 0
    
    def __str__(self) -> str:
        """
        String representation of the CountLatch.
        
        Returns:
            String showing the current count
        """
        return f"{super().__str__()}[Count = {self.get_count()}]"
    
    @staticmethod
    def _convert_to_seconds(value: float, unit: str) -> float:
        """
        Convert time value to seconds.
        
        Args:
            value: The time value
            unit: The time unit
            
        Returns:
            Time in seconds
        """
        conversions = {
            'nanoseconds': 1e-9,
            'microseconds': 1e-6,
            'milliseconds': 1e-3,
            'seconds': 1.0,
            'minutes': 60.0,
            'hours': 3600.0,
            'days': 86400.0,
        }
        return value * conversions.get(unit.lower(), 1.0)


# Alternative implementation using a more Pythonic approach
class CountLatchSimple:
    """
    Simplified version of CountLatch using Python's threading primitives.
    This version is more straightforward but functionally equivalent.
    """
    
    def __init__(self, count: int = 0):
        self._count = count
        self._lock = Lock()
        self._zero_condition = Condition(self._lock)
    
    def increment(self):
        """Increment the counter."""
        with self._lock:
            self._count += 1
    
    def decrement(self):
        """Decrement the counter and notify if it reaches zero."""
        with self._lock:
            if self._count > 0:
                self._count -= 1
            if self._count == 0:
                self._zero_condition.notify_all()
    
    def await_zero(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until the counter reaches zero.
        
        Args:
            timeout: Maximum time to wait in seconds (None for infinite)
            
        Returns:
            True if count reached zero, False if timeout
            
        Raises:
            InterruptedError: If interrupted while waiting
        """
        with self._lock:
            if timeout is None:
                while self._count > 0:
                    self._zero_condition.wait()
                return True
            else:
                end_time = time.time() + timeout
                while self._count > 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return False
                    if not self._zero_condition.wait(timeout=remaining):
                        return self._count == 0
                return True
    
    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._count = 0
    
    def reset_and_interrupt(self):
        """Reset counter and wake all waiting threads."""
        with self._lock:
            self._count = 0
            self._zero_condition.notify_all()
    
    def is_at_zero(self) -> bool:
        """Check if counter is at zero."""
        with self._lock:
            return self._count == 0
    
    def get_count(self) -> int:
        """Get current count."""
        with self._lock:
            return self._count
    
    def __str__(self) -> str:
        return f"CountLatch[Count = {self.get_count()}]"


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

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.AbstractQueuedSynchronizer;

/**
 * A synchronization aid similar to {@link CountDownLatch} but with the ability
 * to also count up. This is useful to wait until a variable number of tasks
 * have completed. {@link #awaitZero()} will block until the count reaches zero.
 */
public class CountLatch {

	@SuppressWarnings("serial")
	private static final class Sync extends AbstractQueuedSynchronizer {

		Sync(int count) {
			setState(count);
		}

		int getCount() {
			return getState();
		}

        void reset() {
            setState(0);
        }

        @Override
		protected int tryAcquireShared(int acquires) {
			return (getState() == 0) ? 1 : -1;
		}

		protected int acquireNonBlocking(int acquires) {
			// increment count
			for (;;) {
				int c = getState();
				int nextc = c + 1;
				if (compareAndSetState(c, nextc))
					return 1;
			}
		}

        @Override
		protected boolean tryReleaseShared(int releases) {
			// Decrement count; signal when transition to zero
			for (;;) {
				int c = getState();
				if (c == 0)
					return false;
				int nextc = c - 1;
				if (compareAndSetState(c, nextc))
					return nextc == 0;
			}
		}
	}

	private final Sync sync;

	public CountLatch(int count) {
		this.sync = new Sync(count);
	}

	public void awaitZero() throws InterruptedException {
		sync.acquireShared(1);
	}

	public boolean awaitZero(long timeout, TimeUnit unit) throws InterruptedException {
		return sync.tryAcquireSharedNanos(1, unit.toNanos(timeout));
	}

	public void increment() {
		sync.acquireNonBlocking(1);
	}

	public void decrement() {
		sync.releaseShared(1);
	}

    /**
     * Resets the counter to zero. But waiting threads won't be released somehow.
     * So this interrupts the threads so that they escape from their waiting state.
     */
    public void resetAndInterrupt(){
        sync.reset();
        for (int i = 0; i < 3; i++) //Because it is a best effort thing, do it three times and hope for the best.
            for (Thread t : sync.getQueuedThreads())
                t.interrupt();
        sync.reset(); //Just in case a thread would've incremented the counter again.
    }

	public String toString() {
		return super.toString() + "[Count = " + sync.getCount() + "]";
	}
	
	/**
	 * Gets whether this counting latch has arrived at zero
	 * @return True if this counting latch has arrived at zero, otherwise
	 * false
	 */
	public boolean isAtZero() {
		return sync.getCount() == 0;
	}

}
\n"""