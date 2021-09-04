#coding=utf8
# ========================================================
#   Copyright (C) 2021 All rights reserved.
#   
#   filename : prepare.py
#   author   : ***
#   date     : 2021-09-01
#   desc     : 
# ======================================================== 

import redis
import socket
import hashlib
from conf import *

pika_conf = {
    "bjyt" : {"host" : "10.208.223.68", "port" : 25598, "auth" : "27cd1a3a7ce26487"},
    "zzzc" : {"host" : "10.173.194.217", "port" : 25598, "auth" : "27cd1a3a7ce26487"}
}


class UrlInfoPika:
    def __init__(self):
        jifang = "zzzc"
        host = pika_conf[jifang]["host"]
        port = pika_conf[jifang]["port"]
        auth = pika_conf[jifang]["auth"]
        self.redis = redis.Redis(host=host, port=port, password=auth, db=0)

    def get_md5_by_gnid(self, gnids):
        i = 0
        batch = []
        ret = []
        for gnid in gnids:
            i += 1
            batch.append(gnid)
            if i % 10000 == 0:
                res = self._get_md5_by_gnid(batch)
                ret.extend(res)
                batch = []
        if len(batch) > 0:
            res = self._get_md5_by_gnid(batch)
            ret.extend(res)
        return ret

    def _get_md5_by_gnid(self, gnids):
        with self.redis.pipeline(transaction=False) as p:
            for gnid in gnids:
                p.get(gnid)
            ret = p.execute()
        return ret



def process_train_asto_md5s(fi):
    gnid_md5 = {}
    gnid_headers = set()
    with open(fi, "r") as ifp:
        for line in ifp:
            segs  = line.strip().split("\t")
            head  = segs[0]
            first = segs[1]
            if first.startswith("http"):
                gnid = head.strip().split(',')[2]
                gnid = "gnid:%s" %gnid
                gnid_md5.setdefault(gnid, hashlib.md5(first.encode("utf-8")).hexdigest())
            elif first.startswith("gnid:"):
                gnid_headers.add(first)
    # get gnid-md5 by pika
    valid_gnids = set(gnid_md5.keys())
    leftgnids = list(gnid_headers - valid_gnids)
    pika = UrlInfoPika()
    md5s = pika.get_md5_by_gnid(leftgnids)
    assert(len(md5s) == len(leftgnids))
    valid_pika_gnids = list(filter(lambda x : x[1], zip(leftgnids, md5s)))
    gnid_md5.update(dict([[x, y.decode("utf-8")] for x, y in valid_pika_gnids]))
    outf = "%s_md5s" %fi
    with open (fi, "r") as ifp, open(outf, "w") as ofp:
        for line in ifp:
            segs  = line.strip().split("\t")
            head  = segs[0]
            first = segs[1]
            md5s = []
            if first.startswith("gnid:"):
                if first in gnid_md5:
                    md5s.append(gnid_md5[first])
                    md5s.extend([hashlib.md5(x.encode("utf-8")).hexdigest() for x in segs[2:]])
                elif len(segs) >= 4:
                    md5s = [hashlib.md5(x.encode("utf-8")).hexdigest() for x in segs[2:]]
            else:
                md5s = [hashlib.md5(x.encode("utf-8")).hexdigest() for x in segs[1:]]
            if len(md5s) >= MIN_SESSION_LENGTH:
                md5s = md5s[0:TRAIN_MAX_SEQ_LEN]
                ofp.write("%s\n" %("\t".join(md5s)))


if __name__ == "__main__":
    process_train_asto_md5s("../data/train")
