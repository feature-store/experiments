import time
from collections import defaultdict
import numpy as np
from enum import Enum

class Policy(Enum):
    TOTAL_ERROR_COLD = 'total_error_cold'
    TOTAL_ERROR = 'total_error'
    LAST_QUERY = 'last_query'
    MAX_PENDING = 'max_pending'
    MIN_PAST = 'min_past'
    ROUND_ROBIN = 'round_robin'
    QUERY_PROPORTIONAL = 'query_proportional'
    BATCH = 'batch'

class UserEventQueue: 
    
    """
    Event queue that selects group of user updates
    (note: we can have another implementation which triggers a bunch of updates together)
    """
    
    def __init__(self, num_keys, policy, past_updates): 
        self.policy = policy 
        self.num_keys = num_keys
        
        # metric tracking
        self.total_error = np.zeros((num_keys))
        self.past_updates = past_updates
        self.queue = defaultdict(list)
        self.staleness = defaultdict(lambda: 0)
        self.last_key = defaultdict(lambda: 0)

        # new baselines 
        self.past_queries = defaultdict(lambda: 0) 
        
    def push(self, key, error_score): 
        
        # calcualte error 

        if key not in self.past_updates and self.policy == "total_error_cold": # unseen keys start with high total error
            self.total_error[key] = 1000000
        else:
            # update per key 
            self.total_error[key] += error_score
        self.queue[key].append((key, error_score))

        # TODO: try moving existing keys to front? 
        self.last_key[key] = time.time()
        self.past_queries[key] += 1
        
    def arg_max(self, data_dict): 
        max_key = None
        max_val = None
        valid_keys = 0
        for key in self.queue.keys():
            
            # cannot select empty queue 
            if len(self.queue[key]) <= 0: 
                continue 
                
            valid_keys += 1
            value = data_dict[key]
            if max_key is None or max_val <= value: 
                assert key is not None, f"Invalid key {data_dict}"
                max_key = key
                max_val = value
        return max_key
        
        
    def choose_key(self):
        if self.policy == Policy.TOTAL_ERROR_COLD or self.policy == Policy.TOTAL_ERROR:
            key = self.arg_max(self.total_error)
        elif self.policy == Policy.LAST_QUERY:
            key = self.arg_max(self.last_key)
        elif self.policy == Policy.MAX_PENDING:
            key = self.arg_max({key: len(self.queue[key]) for key in self.queue.keys()})
        elif self.policy == Policy.MIN_PAST: 
            key = self.arg_max({key: 1/(self.past_updates.setdefault(key, 0)+1) for key in self.queue.keys()})
        elif self.policy == Policy.ROUND_ROBIN: 
            key = self.arg_max(self.staleness)
        elif self.policy == Policy.QUERY_PROPORTIONAL: 
            key = self.arg_max(self.past_queries)
        elif self.policy == Policy.BATCH:
            key = self.arg_max(self.staleness) # doensn't matter
        else: 
            raise ValueError("Invalid policy")
       
        assert key is not None or sum([len(v) for v in self.queue.values()]) == 0, f"Key is none, {self.queue}"
        return key 
    
    def pop(self): 
        key = self.choose_key()
        #print(key, score, self.past_updates.setdefault(key, 0))
        if key is None:
            #print("no updates", self.queue)
            return None 
        events = self.queue[key]
        self.queue[key] = []

        # update metrics 
        for k in self.queue.keys():
            self.staleness[k] += 1
        self.staleness[key] = 0
        self.total_error[key] = 0
        
        # TODO: this is wrong
        self.past_updates[key] = self.past_updates.setdefault(key, 0) + len(events)
            
        return key 

    def size(self): 
        size = 0
        for key in self.queue.keys(): 
            if len(self.queue[key]) > 0:
                size += 1
        return size