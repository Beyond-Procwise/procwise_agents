# ProcWise/engines/query_engine.py

import pandas as pd



class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        self.agent_nick = agent_nick

