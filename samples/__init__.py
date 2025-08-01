from typing import Optional

import hashlib
import http.client
import pathlib
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request

from refinery.units.crypto.cipher.aes import aes


class SampleStore:
    lock = threading.Lock()
    temp = tempfile.TemporaryDirectory(prefix='binary-refinery.test-data.')
    root = pathlib.Path(temp.name)

    def __init__(self):
        self.wait = 0.1

    def _download(self, sha256hash: str, timeout: int = 80):
        def tobytearray(r):
            if isinstance(r, bytearray):
                return r
            return bytearray(r)
        remaining = timeout
        wait = self.wait
        backoff = 0
        req = F'https://github.com/binref/refinery-test-data/blob/master/{sha256hash}.enc?raw=true'
        while remaining > 0:
            clock = time.thread_time()
            try:
                with urllib.request.urlopen(req, timeout=remaining) as response:
                    encoded_sample = tobytearray(response.read())
            except (http.client.RemoteDisconnected, socket.timeout, urllib.error.HTTPError):
                time.sleep(wait)
                wait *= 2
                backoff += 1
            else:
                if not backoff:
                    wait = max(0.1, wait / 2)
                self.wait = wait
                return encoded_sample
            remaining -= time.thread_time() - clock
        raise LookupError(F'Timeout exceeded while looking for {sha256hash}, backed off {backoff} times.')

    def get(self, sha256hash: str, key: Optional[str] = None):
        sha256hash = sha256hash.lower()
        path = self.root / F'{sha256hash}.enc'
        key = key or 'REFINERYTESTDATA'
        key = key.encode('latin1')
        with self.lock:
            try:
                with path.open('rb') as fd:
                    encoded_data = fd.read()
            except FileNotFoundError:
                encoded_data = self._download(sha256hash)
            result = encoded_data | aes(mode='CBC', key=key) | bytearray
            checksum = hashlib.sha256(result).hexdigest().lower()
            if not result or checksum != sha256hash:
                raise ValueError(F'The sample {sha256hash} did not decode correctly with key {key}.')
            with path.open('wb') as fd:
                fd.write(encoded_data)
            return result

    def __getitem__(self, sha256hash: str):
        return self.get(sha256hash)
