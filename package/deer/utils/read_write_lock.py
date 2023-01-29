import threading
import logging

class ReadWriteLock:
	""" A lock object that allows many simultaneous "read locks", but
	only one "write lock." """

	def __init__(self, withPromotion=False):
		self._read_ready = threading.Condition(threading.RLock(	))
		self._readers = 0
		self._writers = 0
		self._promote = withPromotion
		self._readerList = []	# List of Reader thread IDs
		self._writerList = []	# List of Writer thread IDs

	def acquire_read(self):
		logging.debug("RWL : acquire_read()")
		""" Acquire a read lock. Blocks only if a thread has
		acquired the write lock. """
		self._read_ready.acquire(	)
		try:
			while self._writers > 0:
				self._read_ready.wait()
			self._readers += 1
		finally:
			self._readerList.append(threading.get_ident())
			self._read_ready.release(	)

	def release_read(self):
		logging.debug("RWL : release_read()")
		""" Release a read lock. """
		self._read_ready.acquire(	)
		try:
			self._readers -= 1
			if not self._readers:
				self._read_ready.notifyAll(	)
		finally:
			self._readerList.remove(threading.get_ident())
			self._read_ready.release(	)

	def acquire_write(self):
		logging.debug("RWL : acquire_write()")
		""" Acquire a write lock. Blocks until there are no
		acquired read or write locks. """
		self._read_ready.acquire(	)	 # A re-entrant lock lets a thread re-acquire the lock
		self._writers += 1
		self._writerList.append(threading.get_ident())
		while self._readers > 0:
			# promote to write lock, only if all the readers are trying to promote to writer
			# If there are other reader threads, then wait till they complete reading
			if self._promote and threading.get_ident() in self._readerList and set(self._readerList).issubset(set(self._writerList)):
				break
			else:
				self._read_ready.wait(	)

	def release_write(self):
		logging.debug("RWL : release_write()")
		""" Release a write lock. """
		self._writers -= 1
		self._writerList.remove(threading.get_ident())
		self._read_ready.notifyAll(	)
		self._read_ready.release(	)
