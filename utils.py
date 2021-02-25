import os
import pathlib
import jsonpickle
import hashlib
import base64

import portalocker

def get_n_lines(fname):
    if not os.path.exists(fname):
        return 0
    else:
        with open(fname,'r') as fh:
            return sum(1 for line in fh)

def exclusive_write_line(fname,line,max_lines=None):
    if not os.path.exists(os.path.dirname(fname)):
        pathlib.Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(fname, mode='a+') as fh:
        n_lines_in_files=sum(1 for line in fh)
        if max_lines is not None and n_lines_in_files>=max_lines:
            print('max lines ('+str(max_lines) + ') in ' + fname + ' reached, not writing.')
        else:
            fh.write(line+'\n')
            fh.flush()
        os.fsync(fh.fileno())

def hash_dict(dct):
    serialized_dct = jsonpickle.encode(dct)
    check_sum = hashlib.md5().hexdigest()
    hasher = hashlib.sha1(serialized_dct.encode('utf-8'))
    return base64.urlsafe_b64encode(hasher.digest())[:10].decode('ascii')